import torch
import torch.nn as nn


class ConditioningEncoder(nn.Module):
    def __init__(self, vae=None, use_normal_maps=True, use_silhouette_maps=True):
        super().__init__()
        self.vae = vae
        self.use_normal_maps = use_normal_maps
        self.use_silhouette_maps = use_silhouette_maps

    @torch.no_grad()
    def encode_image(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    @torch.no_grad()
    def encode_condition(self, smplx_normal=None, smplx_silhouette=None):
        latents = []
        if self.use_normal_maps and smplx_normal is not None:
            latents.append(self.encode_image(smplx_normal))
        if self.use_silhouette_maps and smplx_silhouette is not None:
            latents.append(self.encode_image(smplx_silhouette))
        if len(latents) == 0:
            return None
        return torch.cat(latents, dim=1)

    @torch.no_grad()
    def build_condition(self, image=None, smplx_normal=None, smplx_silhouette=None):
        latents = []
        if image is not None:
            latents.append(self.encode_image(image))
        condition = self.encode_condition(smplx_normal, smplx_silhouette)
        if condition is not None:
            latents.append(condition)
        if len(latents) == 0:
            return None
        return torch.cat(latents, dim=1)

    @torch.no_grad()
    def decode(self, latents):
        latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents).sample
