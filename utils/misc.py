import torch
from torch import nn


def nograd_param(x):
    """
    Naively make tensor from x, then wrap with nn.Parameter without gradient.
    """
    return nn.Parameter(torch.tensor(x), requires_grad=False)
