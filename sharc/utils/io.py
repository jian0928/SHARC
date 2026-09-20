import os

import numpy as np
import torch


def load_mesh(path):
    import trimesh
    mesh = trimesh.load(path, process=False)
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).float()
    faces = torch.from_numpy(np.asarray(mesh.faces)).long()
    return {'vertices': vertices, 'faces': faces}


def save_mesh(path, vertices, faces, vertex_colors=None):
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
    if vertex_colors is not None:
        mesh.visual.vertex_colors = vertex_colors.detach().cpu().numpy()
    mesh.export(path)


def load_image(path):
    import cv2
    image = cv2.imread(path)[:, :, ::-1]
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def save_image(path, image):
    import cv2
    image_np = (image.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    cv2.imwrite(path, image_np[:, :, ::-1])
