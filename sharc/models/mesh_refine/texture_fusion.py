import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewConsistentTextureFusion(nn.Module):
    def __init__(self, renderer=None, num_steps=None, lr=None, color_weight=None, device=None):
        super().__init__()
        self.renderer = renderer
        self.num_steps = num_steps
        self.lr = lr
        self.color_weight = color_weight
        self.device = device

    def _project_colors(self, vertices, target_colors, viewpoints):
        projected = []
        for i, view in enumerate(viewpoints):
            self.renderer.set_camera(**view)
            pos_clip = self.renderer.project(vertices)
            w = pos_clip[..., 3:4]
            ndc = pos_clip[..., :3] / w
            px = ((ndc[..., 0] + 1.0) * 0.5 * self.renderer.width).long().clamp(0, self.renderer.width - 1)
            py = ((1.0 - (ndc[..., 1] + 1.0) * 0.5) * self.renderer.height).long().clamp(0, self.renderer.height - 1)
            sampled = target_colors[i][0, :, py, px].permute(1, 0)
            projected.append(sampled)
        return torch.stack(projected, dim=0)

    def _init_colors(self, vertices, target_colors, viewpoints, visibility_masks=None):
        projected = self._project_colors(vertices, target_colors, viewpoints)
        if visibility_masks is not None:
            masks = torch.stack(visibility_masks, dim=0).float().to(self.device)
            denom = masks.sum(dim=0, keepdim=True)
            init = (projected * masks).sum(dim=0) / torch.clamp(denom, min=1e-8)
        else:
            init = projected.mean(dim=0)
        return init

    def _compute_loss(self, vertices, faces, vertex_colors, target_colors, viewpoints, visibility_masks=None):
        loss = torch.zeros((), device=self.device)
        for i, view in enumerate(viewpoints):
            self.renderer.set_camera(**view)
            pred_color = self.renderer.render_color(vertices, faces, vertex_colors)
            if visibility_masks is not None:
                mask = visibility_masks[i]
                pred_color = pred_color * mask
                target = target_colors[i].permute(0, 2, 3, 1) * mask
            else:
                target = target_colors[i].permute(0, 2, 3, 1)
            loss = loss + self.color_weight * F.mse_loss(pred_color, target)
        return loss

    def fuse(self, mesh, target_colors, viewpoints, visibility_masks=None):
        vertices = mesh['vertices'].to(self.device)
        faces = mesh['faces'].to(self.device)
        vertex_colors = self._init_colors(vertices, target_colors, viewpoints, visibility_masks).requires_grad_(True)
        optimizer = torch.optim.Adam([vertex_colors], lr=self.lr)
        for _ in range(self.num_steps):
            optimizer.zero_grad()
            loss = self._compute_loss(vertices, faces, vertex_colors, target_colors, viewpoints, visibility_masks)
            loss.backward()
            optimizer.step()
        return {'vertices': vertices, 'faces': faces, 'vertex_colors': vertex_colors.detach()}
