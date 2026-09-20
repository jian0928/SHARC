import torch.nn as nn

from .global_branch import GlobalBranch
from .local_branch import LocalBranch
from .cross_branch_alignment import CrossBranchFeatureAlignment
from .conditioning import ConditioningEncoder


class CrossDetailMultiPerspectiveSynthesizer(nn.Module):
    def __init__(self, global_branch=None, local_branch=None, alignment=None, conditioning=None):
        super().__init__()
        self.global_branch = global_branch if global_branch is not None else GlobalBranch()
        self.local_branch = local_branch if local_branch is not None else LocalBranch()
        self.alignment = alignment if alignment is not None else CrossBranchFeatureAlignment()
        self.conditioning = conditioning if conditioning is not None else ConditioningEncoder()

    def enable_cross_branch_alignment(self, global_module, local_module):
        self.alignment.register(global_module, local_module)

    def disable_cross_branch_alignment(self):
        self.alignment.remove()

    def set_face_mask(self, face_mask):
        self.alignment.set_face_mask(face_mask)

    def forward(self, x_t_global, x_t_local, timestep, encoder_hidden_states=None, condition_latents_global=None, condition_latents_local=None):
        pred_local = self.local_branch(
            x_t_local,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            condition_latents=condition_latents_local,
        )
        pred_global = self.global_branch(
            x_t_global,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            condition_latents=condition_latents_global,
        )
        return pred_global, pred_local
