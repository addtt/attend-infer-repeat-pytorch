import numpy as np
import torch
from torch.utils.data import Dataset


class PyroMultiMNIST(Dataset):
    def __init__(self, path, train):
        self.path = path
        self.train = train
        data = np.load(path, allow_pickle=True)
        x = data['x']
        y = data['y']
        split = 50000
        if train:
            self.x, self.y = x[:split], y[:split]
        else:
            self.x, self.y = x[split:], y[split:]
        
    def __getitem__(self, index):
        """
        Returns (x, y), where x is (1, H, W) in range (0, 1),
        y is a label dict with only a 'n_obj' key.
        """
        # x: uint8, (1, H, W)
        # y: label dict
        x, y = self.x[index], self.y[index]
        y = np.array(len(y))
        x = x / 255.0
        
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        x = x[None]
        y = {'n_obj': y}   # label dict: compatible with multiobject dataloader

        return x, y
        
        
    def __len__(self):
        return len(self.x)
