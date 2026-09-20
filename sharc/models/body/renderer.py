import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr


def compute_vertex_normals(vertices, faces):
    vert_normals = torch.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vert_normals = vert_normals.index_add(0, faces[:, 0], face_normals)
    vert_normals = vert_normals.index_add(0, faces[:, 1], face_normals)
    vert_normals = vert_normals.index_add(0, faces[:, 2], face_normals)
    return F.normalize(vert_normals, dim=-1)


class MeshRenderer:
    def __init__(self, device=None, width=None, height=None, antialias=True):
        self.device = device
        self.width = width
        self.height = height
        self.antialias = antialias
        self.focal = None
        self.principal_point = None
        self.camera_pos = None
        self.camera_lookat = None
        self.camera_up = None
        self.near = None
        self.far = None
        self.ctx = dr.RasterizeGLContext()

    def set_camera(self, focal=None, principal_point=None, camera_pos=None, camera_lookat=None, camera_up=None, near=None, far=None):
        self.focal = focal
        self.principal_point = principal_point
        self.camera_pos = camera_pos
        self.camera_lookat = camera_lookat
        self.camera_up = camera_up
        self.near = near
        self.far = far

    def set_resolution(self, width, height):
        self.width = width
        self.height = height

    def _build_proj_matrix(self, batch_size):
        matrix = torch.zeros((batch_size, 4, 4), device=self.device)
        matrix[:, 0, 0] = self.focal
        matrix[:, 1, 1] = self.focal
        matrix[:, 0, 2] = self.principal_point[0]
        matrix[:, 1, 2] = self.principal_point[1]
        matrix[:, 2, 2] = (self.far + self.near) / (self.near - self.far)
        matrix[:, 2, 3] = 2.0 * self.far * self.near / (self.near - self.far)
        matrix[:, 3, 2] = -1.0
        return matrix

    def _build_view_matrix(self, batch_size):
        eye = self.camera_pos
        target = self.camera_lookat
        up = self.camera_up
        f = F.normalize(target - eye, dim=-1)
        s = F.normalize(torch.cross(f, up, dim=-1), dim=-1)
        u = torch.cross(s, f, dim=-1)
        matrix = torch.zeros((batch_size, 4, 4), device=self.device)
        matrix[:, 0, :3] = s
        matrix[:, 1, :3] = u
        matrix[:, 2, :3] = -f
        matrix[:, 0, 3] = -torch.sum(s * eye, dim=-1)
        matrix[:, 1, 3] = -torch.sum(u * eye, dim=-1)
        matrix[:, 2, 3] = torch.sum(f * eye, dim=-1)
        matrix[:, 3, 3] = 1.0
        return matrix

    def project(self, vertices):
        batch_size = vertices.shape[0]
        mvp = torch.matmul(self._build_proj_matrix(batch_size), self._build_view_matrix(batch_size))
        pos_clip = torch.matmul(vertices, mvp.transpose(1, 2))
        return pos_clip

    def _rasterize(self, vertices, faces):
        pos_clip = self.project(vertices)
        rast, _ = dr.rasterize(self.ctx, pos_clip, faces)
        return rast, pos_clip

    def render_normals(self, vertices, faces, vertex_normals=None):
        rast, pos_clip = self._rasterize(vertices, faces)
        if vertex_normals is None:
            vertex_normals = compute_vertex_normals(vertices, faces)
        normals, _ = dr.interpolate(vertex_normals, rast, faces)
        if self.antialias:
            normals = dr.antialias(normals, rast, pos_clip, faces)
        return F.normalize(normals, dim=-1)

    def render_silhouette(self, vertices, faces):
        rast, _ = self._rasterize(vertices, faces)
        silhouette = (rast[..., 3:] > 0).float()
        return silhouette

    def render_color(self, vertices, faces, vertex_colors):
        rast, pos_clip = self._rasterize(vertices, faces)
        color, _ = dr.interpolate(vertex_colors, rast, faces)
        if self.antialias:
            color = dr.antialias(color, rast, pos_clip, faces)
        return color
