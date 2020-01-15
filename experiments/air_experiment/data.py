import os

from multiobject.pytorch import MultiObjectDataset, MultiObjectDataLoader

from datasets import PyroMultiMNIST

multiobject_paths = {
    'multi_mnist_binary': './data/multi_mnist/multi_binary_mnist_012.npz',
    'multi_dsprites_binary_rgb': './data/multi-dsprites-binary-rgb/multi_dsprites_color_012.npz',
}
multiobject_datasets = multiobject_paths.keys()
pyro_mnist_path = os.path.join('data', 'multi_mnist', 'multi_mnist_pyro.npz')


class DatasetLoader:
    """
    Wrapper for DataLoaders. Data attributes:
    - train: DataLoader object for training set
    - test: DataLoader object for test set
    - data_shape: shape of each data point (channels, height, width)
    - img_size: spatial dimensions of each data point (height, width)
    - color_ch: number of color channels
    """

    def __init__(self, args, cuda):

        # Default arguments for dataloaders
        kwargs = {'num_workers': 1, 'pin_memory': False} if cuda else {}

        # Define training and test set
        if args.dataset_name == 'pyro_multi_mnist':
            train_set = PyroMultiMNIST(pyro_mnist_path, train=True)
            test_set = PyroMultiMNIST(pyro_mnist_path, train=False)
        elif args.dataset_name in multiobject_datasets:
            data_path = multiobject_paths[args.dataset_name]
            train_set = MultiObjectDataset(data_path, train=True)
            test_set = MultiObjectDataset(data_path, train=False)
        else:
            raise RuntimeError("Unrecognized data set '{}'".format(args.dataset_name))

        # Dataloaders
        self.train = MultiObjectDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **kwargs
        )
        self.test = MultiObjectDataLoader(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        self.data_shape = self.train.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]
