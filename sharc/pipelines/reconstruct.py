import torch


class ReconstructionPipeline:
    def __init__(self, synthesizer=None, estimator=None, smplx_model=None, renderer=None, refine_pipeline=None, device=None):
        self.synthesizer = synthesizer
        self.estimator = estimator
        self.smplx_model = smplx_model
        self.renderer = renderer
        self.refine_pipeline = refine_pipeline
        self.device = device

    def estimate_smplx(self, image):
        param_dict = self.estimator.estimate(image)
        return self.estimator.extract_params(param_dict)

    def generate_multiview(self, image, smplx_params, viewpoints):
        normals, colors = self.synthesizer.sample(image, smplx_params, viewpoints)
        silhouettes = self._derive_silhouettes(normals)
        return {'normals': normals, 'colors': colors, 'silhouettes': silhouettes}

    def _derive_silhouettes(self, normals):
        silhouettes = []
        for normal in normals:
            silhouette = (normal.abs().sum(dim=1, keepdim=True) > 0).float()
            silhouettes.append(silhouette)
        return torch.stack(silhouettes, dim=0)

    def __call__(self, image, viewpoints):
        smplx_params = self.estimate_smplx(image)
        multiview = self.generate_multiview(image, smplx_params, viewpoints)
        mesh = self.refine_pipeline(multiview, smplx_params, viewpoints)
        return mesh
