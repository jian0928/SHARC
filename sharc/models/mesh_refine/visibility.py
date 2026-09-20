import numpy as np
import torch
from scipy.spatial import cKDTree


class VisibilityMask:
    def __init__(self, device=None, kdtree_k=None):
        self.device = device
        self.kdtree_k = kdtree_k
        self.tree = None
        self.masks = None

    def build_kdtree(self, vertices):
        self.tree = cKDTree(vertices.detach().cpu().numpy())

    def query_nearest(self, points):
        dist, idx = self.tree.query(points.detach().cpu().numpy(), k=self.kdtree_k)
        idx_t = torch.from_numpy(idx).to(self.device)
        dist_t = torch.from_numpy(dist).to(self.device)
        return idx_t, dist_t

    def _compute_view_mask(self, vertices, renderer, rast, depth_tolerance):
        pos_clip = renderer.project(vertices)
        w = pos_clip[..., 3:4]
        ndc = pos_clip[..., :3] / w
        px = ((ndc[..., 0] + 1.0) * 0.5 * renderer.width).long().clamp(0, renderer.width - 1)
        py = ((1.0 - (ndc[..., 1] + 1.0) * 0.5) * renderer.height).long().clamp(0, renderer.height - 1)
        rast_sample = rast[0, py, px, 3:4]
        inv_w = 1.0 / w
        visible = (rast_sample > 0) & ((rast_sample - inv_w).abs() < depth_tolerance)
        return visible.float()

    def compute_masks(self, vertices, faces, renderer, viewpoints, depth_tolerance=None):
        self.build_kdtree(vertices)
        masks = []
        for view in viewpoints:
            renderer.set_camera(**view)
            rast, _ = renderer._rasterize(vertices, faces)
            mask = self._compute_view_mask(vertices, renderer, rast, depth_tolerance)
            masks.append(mask)
        self.masks = masks
        return masks
