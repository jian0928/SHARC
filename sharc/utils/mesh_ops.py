import torch
import torch.nn.functional as F


def compute_vertex_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals = vertex_normals.index_add(0, faces[:, 0], face_normals)
    vertex_normals = vertex_normals.index_add(0, faces[:, 1], face_normals)
    vertex_normals = vertex_normals.index_add(0, faces[:, 2], face_normals)
    return F.normalize(vertex_normals, dim=-1)


def compute_face_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return F.normalize(normals, dim=-1)


def compute_edge_normals(vertices, faces):
    face_normals = compute_face_normals(vertices, faces)
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    edges = torch.sort(edges, dim=1).values
    unique_edges, inverse = torch.unique(edges, dim=0, return_inverse=True)
    repeated = torch.cat([face_normals, face_normals, face_normals], dim=0)
    edge_normals = torch.zeros((unique_edges.shape[0], 3), device=vertices.device)
    edge_normals = edge_normals.index_add(0, inverse, repeated)
    return F.normalize(edge_normals, dim=-1), unique_edges
