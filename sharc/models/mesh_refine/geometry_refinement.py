import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryRefinement(nn.Module):
    def __init__(self, renderer=None, num_steps=None, lr=None, normal_weight=None, silhouette_weight=None, smoothness_weight=None, edge_normal_weight=None, device=None):
        super().__init__()
        self.renderer = renderer
        self.num_steps = num_steps
        self.lr = lr
        self.normal_weight = normal_weight
        self.silhouette_weight = silhouette_weight
        self.smoothness_weight = smoothness_weight
        self.edge_normal_weight = edge_normal_weight
        self.device = device

    @staticmethod
    def _face_normals(vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        return F.normalize(normals, dim=-1)

    @staticmethod
    def _build_edges(faces):
        edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
        edges = torch.sort(edges, dim=1).values
        unique_edges, inverse = torch.unique(edges, dim=0, return_inverse=True)
        return unique_edges, inverse

    def _laplacian_smoothness(self, vertices, faces):
        edges, _ = self._build_edges(faces)
        laplacian = torch.zeros_like(vertices)
        laplacian = laplacian.index_add(0, edges[:, 0], vertices[edges[:, 1]] - vertices[edges[:, 0]])
        laplacian = laplacian.index_add(0, edges[:, 1], vertices[edges[:, 0]] - vertices[edges[:, 1]])
        return (laplacian ** 2).mean()

    def _edge_normal_loss(self, vertices, faces):
        face_normals = self._face_normals(vertices, faces)
        edges, inverse = self._build_edges(faces)
        repeated = torch.cat([face_normals, face_normals, face_normals], dim=0)
        edge_normals = torch.zeros((edges.shape[0], 3), device=vertices.device)
        edge_normals = edge_normals.index_add(0, inverse, repeated)
        edge_normals = F.normalize(edge_normals, dim=-1)
        edge_mean = edge_normals.mean(dim=0, keepdim=True)
        return ((edge_normals - edge_mean) ** 2).mean()

    def _compute_loss(self, vertices, faces, target_normals, target_silhouettes, viewpoints):
        loss = torch.zeros((), device=self.device)
        for i, view in enumerate(viewpoints):
            self.renderer.set_camera(**view)
            pred_normals = self.renderer.render_normals(vertices, faces)
            pred_silhouette = self.renderer.render_silhouette(vertices, faces)
            loss = loss + self.normal_weight * F.mse_loss(pred_normals, target_normals[i])
            loss = loss + self.silhouette_weight * F.mse_loss(pred_silhouette, target_silhouettes[i])
        loss = loss + self.smoothness_weight * self._laplacian_smoothness(vertices, faces)
        loss = loss + self.edge_normal_weight * self._edge_normal_loss(vertices, faces)
        return loss

    def refine(self, init_mesh, target_normals, target_silhouettes, viewpoints):
        vertices = init_mesh['vertices'].clone().to(self.device).requires_grad_(True)
        faces = init_mesh['faces'].to(self.device)
        optimizer = torch.optim.Adam([vertices], lr=self.lr)
        for _ in range(self.num_steps):
            optimizer.zero_grad()
            loss = self._compute_loss(vertices, faces, target_normals, target_silhouettes, viewpoints)
            loss.backward()
            optimizer.step()
        return {'vertices': vertices.detach(), 'faces': faces}
