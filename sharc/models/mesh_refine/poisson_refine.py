import torch


class PoissonRefine:
    def __init__(self, depth=None):
        self.depth = depth

    def refine(self, vertices, normals=None):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals.detach().cpu().numpy())
        else:
            pcd.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=self.depth)
        return mesh


class SMPLXOffsetEnhance:
    def __init__(self, weight=None, device=None):
        self.weight = weight
        self.device = device

    def enhance(self, vertices, smplx_vertices, region_mask=None):
        offset = smplx_vertices - vertices
        if region_mask is not None:
            offset = offset * region_mask.to(self.device)
        return vertices + self.weight * offset
