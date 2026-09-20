import torch.nn as nn


class JointDenoisingLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, pred_global, pred_global_cbfa, pred_local, noise_global, noise_local, weight_global=None, weight_cbfa=None, weight_local=None):
        loss_global = self.criterion(pred_global, noise_global)
        loss_cbfa = self.criterion(pred_global_cbfa, noise_global)
        loss_local = self.criterion(pred_local, noise_local)
        if weight_global is not None:
            loss_global = loss_global * weight_global
        if weight_cbfa is not None:
            loss_cbfa = loss_cbfa * weight_cbfa
        if weight_local is not None:
            loss_local = loss_local * weight_local
        return loss_global + loss_cbfa + loss_local
