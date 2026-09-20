import numpy as np
import torch
import torch.nn.functional as F


class ToTensor:
    def __call__(self, image):
        tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float()
        return tensor / 255.0


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if self.mean is None or self.std is None:
            return tensor
        mean = torch.as_tensor(self.mean, device=tensor.device).view(-1, 1, 1)
        std = torch.as_tensor(self.std, device=tensor.device).view(-1, 1, 1)
        return (tensor - mean) / std


class Resize:
    def __init__(self, size=None):
        self.size = size

    def __call__(self, tensor):
        if self.size is None:
            return tensor
        return F.interpolate(tensor.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
