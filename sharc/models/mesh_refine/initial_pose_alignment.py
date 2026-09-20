import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialPoseAlignment(nn.Module):
    def __init__(self, smplx_model=None, renderer=None, num_steps=None, lr=None, normal_weight=None, silhouette_weight=None, view_confidence=None, device=None):
        super().__init__()
        self.smplx_model = smplx_model
        self.renderer = renderer
        self.num_steps = num_steps
        self.lr = lr
        self.normal_weight = normal_weight
        self.silhouette_weight = silhouette_weight
        self.view_confidence = view_confidence
        self.device = device

    def _init_optim_params(self, init_params):
        optim_params = {}
        for key in ('betas', 'global_orient', 'body_pose', 'transl'):
            if init_params is not None and key in init_params and init_params[key] is not None:
                optim_params[key] = init_params[key].clone().to(self.device).requires_grad_(True)
        return optim_params

    def _compute_loss(self, vertices, target_normals, target_silhouettes, viewpoints):
        faces = self.smplx_model.faces
        loss = torch.zeros((), device=self.device)
        for i, view in enumerate(viewpoints):
            self.renderer.set_camera(**view)
            pred_normals = self.renderer.render_normals(vertices, faces)
            pred_silhouette = self.renderer.render_silhouette(vertices, faces)
            weight = self.view_confidence[i] if self.view_confidence is not None else 1.0
            loss = loss + weight * self.normal_weight * F.mse_loss(pred_normals, target_normals[i])
            loss = loss + weight * self.silhouette_weight * F.mse_loss(pred_silhouette, target_silhouettes[i])
        return loss

    def align(self, target_normals, target_silhouettes, viewpoints, init_params=None):
        optim_params = self._init_optim_params(init_params)
        optimizer = torch.optim.Adam(optim_params.values(), lr=self.lr)
        for _ in range(self.num_steps):
            optimizer.zero_grad()
            output = self.smplx_model(**optim_params)
            loss = self._compute_loss(output.vertices, target_normals, target_silhouettes, viewpoints)
            loss.backward()
            optimizer.step()
        return {key: value.detach() for key, value in optim_params.items()}
