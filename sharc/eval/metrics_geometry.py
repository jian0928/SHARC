import numpy as np
import torch
from scipy.spatial import cKDTree


def chamfer_distance(pred_vertices, gt_vertices):
    pred_np = pred_vertices.detach().cpu().numpy()
    gt_np = gt_vertices.detach().cpu().numpy()
    gt_tree = cKDTree(gt_np)
    pred_tree = cKDTree(pred_np)
    d1, _ = gt_tree.query(pred_np)
    d2, _ = pred_tree.query(gt_np)
    return float((d1.mean() + d2.mean()) / 2.0)


def point_to_surface(pred_vertices, gt_vertices, gt_faces=None):
    pred_np = pred_vertices.detach().cpu().numpy()
    if gt_faces is not None:
        import trimesh
        mesh = trimesh.Trimesh(vertices=gt_vertices.detach().cpu().numpy(), faces=gt_faces.detach().cpu().numpy())
        _, dist, _ = trimesh.proximity.closest_point(mesh, pred_np)
        return float(dist.mean())
    gt_tree = cKDTree(gt_vertices.detach().cpu().numpy())
    dist, _ = gt_tree.query(pred_np)
    return float(dist.mean())


def normal_consistency(pred_vertices, pred_normals, gt_vertices, gt_normals):
    pred_np = pred_vertices.detach().cpu().numpy()
    gt_np = gt_vertices.detach().cpu().numpy()
    tree = cKDTree(gt_np)
    _, idx = tree.query(pred_np)
    pred_n = pred_normals.detach().cpu().numpy()
    gt_n = gt_normals.detach().cpu().numpy()
    dot = (pred_n * gt_n[idx]).sum(axis=1)
    norms = np.linalg.norm(pred_n, axis=1) * np.linalg.norm(gt_n[idx], axis=1)
    return float((dot / norms).mean())


def f_score(pred_vertices, gt_vertices, threshold=None):
    pred_np = pred_vertices.detach().cpu().numpy()
    gt_np = gt_vertices.detach().cpu().numpy()
    gt_tree = cKDTree(gt_np)
    pred_tree = cKDTree(pred_np)
    d1, _ = gt_tree.query(pred_np)
    d2, _ = pred_tree.query(gt_np)
    precision = float((d2 < threshold).mean())
    recall = float((d1 < threshold).mean())
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
