import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class GlobalBranch(nn.Module):
    def __init__(self, unet_pretrained_path=None, unet_config=None, extra_in_channels=None):
        super().__init__()
        if unet_pretrained_path is not None:
            self.unet = UNet2DConditionModel.from_pretrained(unet_pretrained_path, subfolder='unet')
        else:
            self.unet = UNet2DConditionModel(**unet_config)
        self.extra_in_channels = extra_in_channels
        if extra_in_channels is not None:
            self._expand_input_channels(extra_in_channels)

    def _expand_input_channels(self, extra_in_channels):
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            in_channels=old_conv.in_channels + extra_in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding,
        )
        with torch.no_grad():
            new_conv.weight[:, : old_conv.in_channels] = old_conv.weight
        self.unet.conv_in = new_conv

    def forward(self, sample, timestep, encoder_hidden_states=None, condition_latents=None, **kwargs):
        if self.extra_in_channels is not None and condition_latents is not None:
            sample = torch.cat([sample, condition_latents], dim=1)
        return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs).sample
