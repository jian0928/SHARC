import argparse
import math
import os

import torch

from sharc.models.body import PIXIEEstimator, SMPLXModel
from sharc.models.body.renderer import MeshRenderer
from sharc.models.mesh_refine import (
    GeometryRefinement,
    InitialPoseAlignment,
    TopologyAwareColorCompletion,
    ViewConsistentTextureFusion,
    VisibilityMask,
)
from sharc.models.synthesizer import CrossDetailMultiPerspectiveSynthesizer
from sharc.pipelines import MeshRefinePipeline, ReconstructionPipeline
from sharc.utils.io import load_image, save_mesh
from sharc.utils.seed import set_seed


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
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out', type=str, default='output')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
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

    set_seed(args.seed)
    smplx_model = SMPLXModel(model_path=args.model_path)
    renderer = MeshRenderer(device=args.device, width=args.width, height=args.height)
    viewpoints = [build_view(angle, args) for angle in args.angles]
    ipa = InitialPoseAlignment(smplx_model=smplx_model, renderer=renderer, num_steps=args.ipa_steps, lr=args.ipa_lr, device=args.device)
    gdr = GeometryRefinement(renderer=renderer, num_steps=args.gdr_steps, lr=args.gdr_lr, device=args.device)
    vctf = ViewConsistentTextureFusion(renderer=renderer, num_steps=args.vctf_steps, lr=args.vctf_lr, device=args.device)
    visibility = VisibilityMask(device=args.device)
    color_completion = TopologyAwareColorCompletion(device=args.device)
    refine_pipeline = MeshRefinePipeline(
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
    estimator = PIXIEEstimator(device=args.device)
    synthesizer = CrossDetailMultiPerspectiveSynthesizer()
    pipeline = ReconstructionPipeline(
        synthesizer=synthesizer,
        estimator=estimator,
        smplx_model=smplx_model,
        renderer=renderer,
        refine_pipeline=refine_pipeline,
        device=args.device,
    )
    image = load_image(args.image).unsqueeze(0).to(args.device)
    textured = pipeline(image, viewpoints)
    os.makedirs(args.out, exist_ok=True)
    save_mesh(os.path.join(args.out, 'textured.obj'), textured['vertices'], textured['faces'], textured.get('vertex_colors'))


if __name__ == '__main__':
    main()
