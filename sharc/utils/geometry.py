import numpy as np
import torch
from scipy.spatial import cKDTree


def build_kdtree(points):
    return cKDTree(points.detach().cpu().numpy())


def nearest_neighbors(tree, points, k=None):
    dist, idx = tree.query(points.detach().cpu().numpy(), k=k)
    return torch.from_numpy(idx), torch.from_numpy(dist)


def chamfer_distance(points_a, points_b):
    tree_a = build_kdtree(points_a)
    tree_b = build_kdtree(points_b)
    d1, _ = tree_b.query(points_a.detach().cpu().numpy())
    d2, _ = tree_a.query(points_b.detach().cpu().numpy())
    return float((d1.mean() + d2.mean()) / 2.0)
