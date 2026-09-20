import smplx
import torch.nn as nn


class SMPLXModel(nn.Module):
    def __init__(self, model_path, gender='neutral', use_pca=True, num_pca_comps=None, flat_hand_mean=False, batch_size=None, **kwargs):
        super().__init__()
        self.model = smplx.create(
            model_path=model_path,
            model_type='smplx',
            gender=gender,
            use_pca=use_pca,
            num_pca_comps=num_pca_comps,
            flat_hand_mean=flat_hand_mean,
            batch_size=batch_size,
            **kwargs,
        )

    def forward(self, betas=None, global_orient=None, body_pose=None, left_hand_pose=None, right_hand_pose=None, jaw_pose=None, leye_pose=None, reye_pose=None, transl=None, return_verts=True, return_full_pose=False):
        return self.model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            transl=transl,
            return_verts=return_verts,
            return_full_pose=return_full_pose,
        )

    def get_vertices(self, betas=None, global_orient=None, body_pose=None, left_hand_pose=None, right_hand_pose=None, jaw_pose=None, transl=None):
        output = self.forward(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            transl=transl,
        )
        return output.vertices

    @property
    def faces(self):
        return self.model.faces

    @property
    def v_template(self):
        return self.model.v_template

    @property
    def num_verts(self):
        return self.v_template.shape[0]
