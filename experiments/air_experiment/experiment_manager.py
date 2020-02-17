import argparse
import os

import torch
import torch.nn.functional as F
from boilr import VIExperimentManager
from boilr.viz import img_grid_pad_value
from torch import optim
from torchvision.utils import save_image

from models.air import AIR
from utils.spatial_transform import batch_add_bounding_boxes
from .data import DatasetLoader


class AIRExperiment(VIExperimentManager):
    """
    Experiment manager.

    Data attributes:
    - 'args': argparse.Namespace containing all config parameters. When
      initializing this object, if 'args' is not given, all config
      parameters are set based on experiment defaults and user input, using
      argparse.
    - 'run_description': string description of the run that includes a timestamp
      and can be used e.g. as folder name for logging.
    - 'model': the model.
    - 'device': torch.device that is being used
    - 'dataloaders': DataLoaders, with attributes 'train' and 'test'
    - 'optimizer': the optimizer
    """

    def make_datamanager(self):
        cuda = self.device.type == 'cuda'
        return DatasetLoader(self.args, cuda)

    def make_model(self):
        args = self.args

        obj_size = {
            'pyro_multi_mnist': 28,
            'multi_mnist_binary': 18,
            'multi_dsprites_binary_rgb': 18,
        }[args.dataset_name]

        scale_prior_mean = {
            'pyro_multi_mnist': 3.,
            'multi_mnist_binary': 4.5,
            'multi_dsprites_binary_rgb': 4.5,
        }[args.dataset_name]

        model = AIR(
            img_size=self.dataloaders.img_size[0],  # assume h=w
            color_channels=self.dataloaders.color_ch,
            object_size=obj_size,
            max_steps=3,
            likelihood=args.likelihood,
            scale_prior_mean=scale_prior_mean,
        )
        return model

    def make_optimizer(self):
        args = self.args
        optimizer = optim.Adam([
            {
                'params': self.model.air_params(),
                'lr': args.lr,
                'weight_decay': args.weight_decay,
            },
            {
                'params': self.model.baseline_params(),
                'lr': args.bl_lr,
                'weight_decay': args.weight_decay,
            },
        ])
        return optimizer



    def forward_pass(self, x, y=None):
        """
        Simple single-pass model evaluation. It consists of a forward pass
        and computation of all necessary losses and metrics.
        """

        # Forward pass
        x = x.to(self.device, non_blocking=True)
        out = self.model(x)

        elbo_sep = out['elbo_sep']
        bl_target = out['baseline_target']
        bl_value = out['baseline_value']
        data_likelihood_sep = out['data_likelihood']
        z_pres_likelihood = out['z_pres_likelihood']
        mask = out['mask_prev']

        # The baseline target is:
        # sum_{i=t}^T KL[i] - log p(x | z)
        # for all steps up to (and including) the first z_pres=0
        bl_target = bl_target - data_likelihood_sep[:, None]
        bl_target = bl_target * mask  # (B, T)

        # The "REINFORCE" term in the gradient is:
        # (baseline_target - baseline_value) * gradient[z_pres_likelihood]
        reinforce_term = ((bl_target - bl_value).detach() * z_pres_likelihood)
        reinforce_term = reinforce_term * mask
        reinforce_term = reinforce_term.sum(1)   # (B, )

        # Maximize ELBO with additional REINFORCE term for discrete variables
        model_loss = reinforce_term - elbo_sep   # (B, )
        model_loss = model_loss.mean()    # mean over batch

        # MSE as baseline loss
        baseline_loss = F.mse_loss(bl_value, bl_target.detach(), reduction='none')
        baseline_loss = baseline_loss * mask
        baseline_loss = baseline_loss.sum(1).mean()  # mean over batch

        loss = model_loss + baseline_loss
        out['loss'] = loss

        # L2
        l2 = 0.0
        for p in self.model.parameters():
            l2 = l2 + torch.sum(p ** 2)
        l2 = l2.sqrt()
        out['l2'] = l2

        # Accuracy
        out['accuracy'] = None
        if y is not None:
            n_obj = y['n_obj'].to(self.device)
            n_pred = out['inferred_n']  # (B, )
            correct = (n_pred == n_obj).float().sum()
            acc = correct / n_pred.size(0)
            out['accuracy'] = acc

        # TODO Only for viz, as std=0.3 is pretty high so samples are not good
        out['out_sample'] = out['out_mean']  # this is actually NOT a sample!

        return out


    @staticmethod
    def print_train_log(step, epoch, summaries):
        s = ("       [step {step}]   loss: {loss:.5g}   ELBO: {elbo:.5g}   "
             "recons: {recons:.3g}   KL: {kl:.3g}   acc: {acc:.3g}")
        s = s.format(
            step=step,
            loss=summaries['loss/loss'],
            elbo=summaries['elbo/elbo'],
            recons=summaries['elbo/recons'],
            kl=summaries['elbo/kl'],
            acc=summaries['accuracy'])
        print(s)


    @staticmethod
    def print_test_log(summaries, step=None, epoch=None):
        log_string = "       "
        if epoch is not None:
            log_string += "[step {}, epoch {}]   ".format(step, epoch)
        s = "ELBO {elbo:.5g}   recons: {recons:.3g}   KL: {kl:.3g}   acc: {acc:.3g}"
        log_string += s.format(
            elbo=summaries['elbo/elbo'],
            recons=summaries['elbo/recons'],
            kl=summaries['elbo/kl'],
            acc=summaries['accuracy'])
        ll_key = None
        for k in summaries.keys():
            if k.find('elbo_IW') > -1:
                ll_key = k
                iw_samples = k.split('_')[-1]
                break
        if ll_key is not None:
            log_string += "   marginal log-likelihood ({}) {:.5g}".format(
                iw_samples, summaries[ll_key])

        print(log_string)


    @staticmethod
    def get_metrics_dict(results):
        metrics_dict = {
            'loss/loss': results['loss'].item(),
            'elbo/elbo': results['elbo'].item(),
            'elbo/recons': results['recons'].item(),
            'elbo/kl': results['kl'].item(),
            'l2/l2': results['l2'].item(),
            'accuracy': results['accuracy'].item(),

            'kl/pres': results['kl_pres'].item(),
            'kl/what': results['kl_what'].item(),
            'kl/where': results['kl_where'].item(),
        }
        return metrics_dict



    def additional_testing(self, img_folder):
        """
        Perform additional testing, including possibly generating images.

        In this case, save samples from the generative model, and pairs
        input/reconstruction from the test set.

        :param img_folder: folder to store images
        """

        step = self.model.global_step

        if not self.args.dry_run:

            # Saved images will have n**2 sub-images
            n = 8

            # Save model samples
            sample, zwhere, n_obj = self.model.sample_prior(n ** 2)
            annotated_sample = batch_add_bounding_boxes(sample, zwhere, n_obj)
            fname = os.path.join(img_folder, 'sample_' + str(step) + '.png')
            pad = img_grid_pad_value(annotated_sample)
            save_image(annotated_sample, fname, nrow=n, pad_value=pad)

            # Get first test batch
            (x, _) = next(iter(self.dataloaders.test))
            fname = os.path.join(img_folder, 'reconstruction_' + str(step) + '.png')

            # Save model original/reconstructions
            self.save_input_and_recons(x, fname, n)


    def save_input_and_recons(self, x, fname, n):
        n_img = n ** 2 // 2
        if x.shape[0] < n_img:
            msg = ("{} data points required, but given batch has size {}. "
                   "Please use a larger batch.".format(n_img, x.shape[0]))
            raise RuntimeError(msg)
        x = x.to(self.device)
        outputs = self.forward_pass(x)
        x = x[:n_img]
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        recons = outputs['out_sample'][:n_img]
        z_where = outputs['all_z_where'][:n_img]
        pred_count = outputs['inferred_n']
        recons = batch_add_bounding_boxes(recons, z_where, pred_count)
        imgs = torch.stack([x.cpu(), recons.cpu()])
        imgs = imgs.permute(1, 0, 2, 3, 4)
        imgs = imgs.reshape(n ** 2, x.size(1), x.size(2), x.size(3))
        pad = img_grid_pad_value(imgs)
        save_image(imgs, fname, nrow=n, pad_value=pad)


    def _parse_args(self):
        """
        Parse command-line arguments defining experiment settings.

        :return: args: argparse.Namespace with experiment settings
        """

        def list_options(lst):
            if lst:
                return "'" + "' | '".join(lst) + "'"
            return ""

        legal_datasets = [
            'pyro_multi_mnist',
            'multi_mnist_binary',
            'multi_dsprites_binary_rgb']
        legal_likelihoods = ['bernoulli', 'original']

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            allow_abbrev=False)

        self.add_required_args(parser,

                               # General
                               batch_size=64,
                               test_batch_size=2000,
                               lr=1e-4,
                               seed=54321,
                               log_interval=50000,
                               test_log_interval=50000,
                               checkpoint_interval=100000,
                               resume="",

                               # VI-specific
                               ll_every=50000,
                               loglik_samples=100,)

        parser.add_argument('-d', '--dataset',
                            type=str,
                            choices=legal_datasets,
                            default='pyro_multi_mnist',
                            metavar='NAME',
                            dest='dataset_name',
                            help="dataset: " + list_options(legal_datasets))

        parser.add_argument('--likelihood',
                            type=str,
                            choices=legal_likelihoods,
                            metavar='NAME',
                            dest='likelihood',
                            help="likelihood: {}; default depends on dataset".format(
                                list_options(legal_likelihoods)))

        parser.add_argument('--bl-lr',
                            type=float,
                            default=1e-1,
                            metavar='LR',
                            help="baseline's learning rate")

        parser.add_argument('--wd',
                            type=float,
                            default=0.0,
                            dest='weight_decay',
                            help='weight decay')

        args = parser.parse_args()

        assert args.loglik_interval % args.test_log_interval == 0

        if args.likelihood is None:  # defaults
            args.likelihood = {
                'pyro_multi_mnist': 'original',
                'multi_mnist_binary': 'original',  # 'bernoulli',
                'multi_dsprites_binary_rgb': 'original',  # 'bernoulli',
            }[args.dataset_name]

        return args

    @staticmethod
    def _make_run_description(args):
        """
        Create a string description of the run. It is used in the names of the
        logging folders.

        :param args: experiment config
        :return: the run description
        """
        s = ''
        s += args.dataset_name
        s += ',seed{}'.format(args.seed)
        if len(args.additional_descr) > 0:
            s += ',' + args.additional_descr
        return s
