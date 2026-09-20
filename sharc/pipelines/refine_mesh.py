import torch


class MeshRefinePipeline:
    def __init__(self, smplx_model=None, renderer=None, ipa=None, gdr=None, vctf=None, visibility=None, color_completion=None, poisson_refine=None, smplx_offset=None, depth_tolerance=None, device=None):
        self.smplx_model = smplx_model
        self.renderer = renderer
        self.ipa = ipa
        self.gdr = gdr
        self.vctf = vctf
        self.visibility = visibility
        self.color_completion = color_completion
        self.poisson_refine = poisson_refine
        self.smplx_offset = smplx_offset
        self.depth_tolerance = depth_tolerance
        self.device = device

    @staticmethod
    def _split_views(tensor):
        return [tensor[i] for i in range(tensor.shape[0])]

    def align_pose(self, smplx_params, target_normals, target_silhouettes, viewpoints):
        aligned = self.ipa.align(target_normals, target_silhouettes, viewpoints, init_params=smplx_params)
        return aligned

    def refine_geometry(self, init_mesh, target_normals, target_silhouettes, viewpoints):
        return self.gdr.refine(init_mesh, target_normals, target_silhouettes, viewpoints)

    def fuse_texture(self, mesh, target_colors, viewpoints, visibility_masks=None):
        return self.vctf.fuse(mesh, target_colors, viewpoints, visibility_masks)

    def __call__(self, multiview, smplx_params, viewpoints):
        target_normals = self._split_views(multiview['normals'])
        target_colors = self._split_views(multiview['colors'])
        target_silhouettes = self._split_views(multiview['silhouettes'])
        aligned = self.align_pose(smplx_params, target_normals, target_silhouettes, viewpoints)
        output = self.smplx_model(**aligned)
        init_mesh = {'vertices': output.vertices.detach().to(self.device), 'faces': self.smplx_model.faces.to(self.device)}
        refined = self.refine_geometry(init_mesh, target_normals, target_silhouettes, viewpoints)
        visibility_masks = None
        if self.visibility is not None:
            visibility_masks = self.visibility.compute_masks(refined['vertices'], refined['faces'], self.renderer, viewpoints, depth_tolerance=self.depth_tolerance)
        textured = self.fuse_texture(refined, target_colors, viewpoints, visibility_masks)
        if self.color_completion is not None:
            textured['vertex_colors'] = self.color_completion.complete(textured['vertex_colors'], textured['faces'], visibility_masks)
        return textured
