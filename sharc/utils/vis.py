import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_images(images, titles=None, save_path=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    if n == 1:
        axes = [axes]
    for ax, image in zip(axes, images):
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        ax.imshow(image_np)
        ax.axis('off')
    if titles is not None:
        for ax, title in zip(axes, titles):
            ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mesh(vertices, faces, vertex_colors=None, save_path=None):
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
    if vertex_colors is not None:
        mesh.visual.vertex_colors = vertex_colors.detach().cpu().numpy()
    scene = trimesh.Scene(mesh)
    if save_path is not None:
        png = scene.save_image()
        with open(save_path, 'wb') as f:
            f.write(png)
    return scene
