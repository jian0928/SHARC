import torch


class TopologyAwareColorCompletion:
    def __init__(self, device=None):
        self.device = device

    @staticmethod
    def _build_edges(faces):
        edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
        edges = torch.sort(edges, dim=1).values
        unique_edges, _ = torch.unique(edges, dim=0, return_inverse=True)
        return unique_edges

    def complete(self, vertex_colors, faces, visibility_masks=None, num_iters=None):
        colors = vertex_colors.clone().to(self.device)
        if visibility_masks is not None:
            seen = torch.stack(visibility_masks, dim=0).sum(dim=0).squeeze(-1) > 0
        else:
            seen = torch.ones(colors.shape[0], dtype=torch.bool, device=self.device)
        invisible = ~seen
        if not invisible.any():
            return colors
        edges = self._build_edges(faces.to(self.device))
        n = colors.shape[0]
        ones = torch.ones(edges.shape[0], device=self.device)
        for _ in range(num_iters):
            adj_sum = torch.zeros_like(colors)
            adj_count = torch.zeros(n, device=self.device)
            adj_sum = adj_sum.index_add(0, edges[:, 0], colors[edges[:, 1]])
            adj_sum = adj_sum.index_add(0, edges[:, 1], colors[edges[:, 0]])
            adj_count = adj_count.index_add(0, edges[:, 0], ones)
            adj_count = adj_count.index_add(0, edges[:, 1], ones)
            avg = adj_sum / torch.clamp(adj_count.unsqueeze(-1), min=1e-8)
            colors = torch.where(invisible.unsqueeze(-1), avg, colors)
        return colors
