import argparse
import glob
import json
import os

import torch

from sharc.eval import Evaluator
from sharc.utils.io import load_mesh
from sharc.utils.mesh_ops import compute_vertex_normals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default='output')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--thresholds', type=float, nargs='+', default=None)
    args = parser.parse_args()

    evaluator = Evaluator(chamfer_thresholds=args.thresholds, device=args.device)
    results = []
    for pred_path in sorted(glob.glob(os.path.join(args.pred_dir, '*.obj'))):
        name = os.path.basename(pred_path)
        gt_path = os.path.join(args.gt_dir, name)
        if not os.path.exists(gt_path):
            continue
        pred = load_mesh(pred_path)
        gt = load_mesh(gt_path)
        pred_normals = compute_vertex_normals(pred['vertices'], pred['faces'])
        gt_normals = compute_vertex_normals(gt['vertices'], gt['faces'])
        metrics = evaluator.evaluate_geometry(pred['vertices'], pred_normals, gt['vertices'], gt_normals, gt_faces=gt['faces'])
        metrics['name'] = name
        results.append(metrics)
    summary = Evaluator.summarize(results)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
