import argparse
import math
import os

import torch

from sharc.models.body import SMPLXModel
from sharc.models.body.renderer import MeshRenderer
from sharc.models.mesh_refine import (
    GeometryRefinement,
    InitialPoseAlignment,
    TopologyAwareColorCompletion,
    ViewConsistentTextureFusion,
    VisibilityMask,
)
from sharc.pipelines import MeshRefinePipeline
from sharc.utils.io import load_image, save_mesh


def build_view(angle, args):
    rad = math.radians(angle)
    camera_pos = torch.tensor([[args.camera_radius * math.sin(rad), args.camera_height, args.camera_radius * math.cos(rad)]], device=args.device)
    principal_point = torch.tensor(args.principal_point, device=args.device) if args.principal_point is not None else None
    return {
        'focal': args.focal,
        'principal_point': principal_point,
        'camera_pos': camera_pos,
        'camera_lookat': torch.zeros(1, 3, device=args.device),
        'camera_up': torch.tensor([[0.0, 1.0, 0.0]], device=args.device),
        'near': args.near,
        'far': args.far,
    }


def load_multiview(directory, angles):
    normals = []
    colors = []
    silhouettes = []
    for angle in angles:
        normals.append(load_image(os.path.join(directory, 'normal_%d.png' % angle)).unsqueeze(0))
        colors.append(load_image(os.path.join(directory, 'color_%d.png' % angle)).unsqueeze(0))
        silhouettes.append(load_image(os.path.join(directory, 'silhouette_%d.png' % angle)).unsqueeze(0))
    return {
        'normals': torch.stack(normals),
        'colors': torch.stack(colors),
        'silhouettes': torch.stack(silhouettes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiview_dir', type=str, required=True)
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out', type=str, default='output')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--angles', type=float, nargs='+', required=True)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--focal', type=float, default=None)
    parser.add_argument('--principal_point', type=float, nargs=2, default=None)
    parser.add_argument('--camera_radius', type=float, default=None)
    parser.add_argument('--camera_height', type=float, default=None)
    parser.add_argument('--near', type=float, default=None)
    parser.add_argument('--far', type=float, default=None)
    parser.add_argument('--ipa_steps', type=int, default=None)
    parser.add_argument('--ipa_lr', type=float, default=None)
    parser.add_argument('--gdr_steps', type=int, default=None)
    parser.add_argument('--gdr_lr', type=float, default=None)
    parser.add_argument('--vctf_steps', type=int, default=None)
    parser.add_argument('--vctf_lr', type=float, default=None)
    parser.add_argument('--depth_tolerance', type=float, default=None)
    args = parser.parse_args()

    smplx_model = SMPLXModel(model_path=args.model_path)
    renderer = MeshRenderer(device=args.device, width=args.width, height=args.height)
    viewpoints = [build_view(angle, args) for angle in args.angles]
    ipa = InitialPoseAlignment(smplx_model=smplx_model, renderer=renderer, num_steps=args.ipa_steps, lr=args.ipa_lr, device=args.device)
    gdr = GeometryRefinement(renderer=renderer, num_steps=args.gdr_steps, lr=args.gdr_lr, device=args.device)
    vctf = ViewConsistentTextureFusion(renderer=renderer, num_steps=args.vctf_steps, lr=args.vctf_lr, device=args.device)
    visibility = VisibilityMask(device=args.device)
    color_completion = TopologyAwareColorCompletion(device=args.device)
    pipeline = MeshRefinePipeline(
        smplx_model=smplx_model,
        renderer=renderer,
        ipa=ipa,
        gdr=gdr,
        vctf=vctf,
        visibility=visibility,
        color_completion=color_completion,
        depth_tolerance=args.depth_tolerance,
        device=args.device,
    )
    multiview = load_multiview(args.multiview_dir, args.angles)
    smplx_params = torch.load(args.params)
    textured = pipeline(multiview, smplx_params, viewpoints)
    os.makedirs(args.out, exist_ok=True)
    save_mesh(os.path.join(args.out, 'textured.obj'), textured['vertices'], textured['faces'], textured.get('vertex_colors'))


if __name__ == '__main__':
    main()
