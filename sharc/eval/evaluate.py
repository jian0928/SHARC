import numpy as np
import torch

from .metrics_appearance import dists, lpips, psnr, ssim
from .metrics_geometry import chamfer_distance, f_score, normal_consistency, point_to_surface


class Evaluator:
    def __init__(self, chamfer_thresholds=None, device=None):
        self.chamfer_thresholds = chamfer_thresholds
        self.device = device

    def evaluate_geometry(self, pred_vertices, pred_normals, gt_vertices, gt_normals, gt_faces=None):
        metrics = {
            'cd': chamfer_distance(pred_vertices, gt_vertices),
            'p2s': point_to_surface(pred_vertices, gt_vertices, gt_faces),
            'nc': normal_consistency(pred_vertices, pred_normals, gt_vertices, gt_normals),
        }
        if self.chamfer_thresholds is not None:
            for threshold in self.chamfer_thresholds:
                metrics['f_score_%s' % threshold] = f_score(pred_vertices, gt_vertices, threshold=threshold)
        return metrics

    def evaluate_appearance(self, pred_images, gt_images, lpips_model=None, dists_model=None):
        metrics = {
            'psnr': psnr(pred_images, gt_images),
            'ssim': ssim(pred_images, gt_images),
        }
        if lpips_model is not None:
            metrics['lpips'] = lpips(pred_images, gt_images, model=lpips_model)
        if dists_model is not None:
            metrics['dists'] = dists(pred_images, gt_images, model=dists_model)
        return metrics

    @staticmethod
    def summarize(results):
        from scipy import stats
        if len(results) == 0:
            return {}
        keys = list(results[0].keys())
        summary = {}
        for key in keys:
            values = np.array([r[key] for r in results])
            n = len(values)
            mean = values.mean()
            std = values.std(ddof=1) if n > 1 else 0.0
            t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
            ci = t_crit * std / np.sqrt(n)
            summary[key] = {'mean': float(mean), 'std': float(std), 'ci95': float(ci)}
        return summary
