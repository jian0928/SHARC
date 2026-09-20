import argparse
import math
import os

import torch

from sharc.models.body import SMPLXModel
from sharc.models.body.renderer import MeshRenderer
from sharc.utils.io import save_image


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


def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    params = torch.load(args.params)
    body_params = {
        'betas': params.get('betas', params.get('beta')),
        'body_pose': params.get('body_pose', params.get('theta')),
        'global_orient': params.get('global_orient'),
        'transl': params.get('transl'),
        'jaw_pose': params.get('jaw_pose'),
        'leye_pose': params.get('leye_pose'),
        'reye_pose': params.get('reye_pose'),
        'left_hand_pose': params.get('left_hand_pose'),
        'right_hand_pose': params.get('right_hand_pose'),
    }
    smplx_model = SMPLXModel(model_path=args.model_path)
    output = smplx_model(**body_params)
    vertices = output.vertices
    faces = smplx_model.faces
    renderer = MeshRenderer(device=args.device, width=args.width, height=args.height)
    os.makedirs(args.out, exist_ok=True)
    for angle in args.angles:
        renderer.set_camera(**build_view(angle, args))
        normals = renderer.render_normals(vertices, faces)
        silhouette = renderer.render_silhouette(vertices, faces)
        save_image(os.path.join(args.out, 'normal_%d.png' % angle), normals[0].permute(2, 0, 1))
        save_image(os.path.join(args.out, 'silhouette_%d.png' % angle), silhouette[0].permute(2, 0, 1))


if __name__ == '__main__':
    main()
