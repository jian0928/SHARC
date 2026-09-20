import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossBranchFeatureAlignment(nn.Module):
    def __init__(self, enabled=True, layer_index=None):
        super().__init__()
        self.enabled = enabled
        self.layer_index = layer_index
        self.face_mask = None
        self.local_features = None
        self.handles = []

    def set_face_mask(self, face_mask):
        self.face_mask = face_mask

    def capture_local(self, module, input, output):
        self.local_features = output[0] if isinstance(output, tuple) else output

    def fuse_global(self, module, input, output):
        if not self.enabled or self.local_features is None or self.face_mask is None:
            return output
        h_b = output[0] if isinstance(output, tuple) else output
        h_f = self.local_features
        mask = self.face_mask
        if h_f.shape[2:] != h_b.shape[2:]:
            h_f = F.interpolate(h_f, size=h_b.shape[2:], mode='bilinear', align_corners=False)
        if mask.shape[2:] != h_b.shape[2:]:
            mask = F.interpolate(mask, size=h_b.shape[2:], mode='nearest')
        fused = h_b + mask * h_f
        if isinstance(output, tuple):
            return (fused,) + output[1:]
        return fused

    def register(self, global_module, local_module):
        self.handles.append(local_module.register_forward_hook(self.capture_local))
        self.handles.append(global_module.register_forward_hook(self.fuse_global))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def reset(self):
        self.local_features = None
        self.face_mask = None
