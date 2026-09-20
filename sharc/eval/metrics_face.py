import torch
import torch.nn.functional as F


def identity_similarity(pred, target, arcface=None):
    if arcface is None:
        raise ValueError('arcface model required')
    emb_pred = arcface(pred)
    emb_target = arcface(target)
    return float(F.cosine_similarity(emb_pred, emb_target, dim=-1).mean().item())
