import os
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, root=None, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise NotImplementedError
