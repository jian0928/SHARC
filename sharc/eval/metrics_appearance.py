import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred, target, data_range=None):
    if data_range is None:
        data_range = 1.0
    mse = F.mse_loss(pred, target)
    return float((10.0 * torch.log10(data_range ** 2 / mse)).item())


def ssim(pred, target, data_range=None):
    from skimage.metrics import structural_similarity
    pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    target_np = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
    if data_range is None:
        data_range = 1.0
    scores = [structural_similarity(p, t, channel_axis=-1, data_range=data_range) for p, t in zip(pred_np, target_np)]
    return float(np.mean(scores))


def lpips(pred, target, model=None):
    if model is None:
        import lpips
        model = lpips.LPIPS(net='alex')
    return float(model(pred, target).mean().item())


def dists(pred, target, model=None):
    if model is None:
        from dists import DISTS
        model = DISTS()
    return float(model(pred, target).mean().item())
