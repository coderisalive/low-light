"""Lightweight paired-supervised RGB low-light restoration U-Net.

The network predicts a residual in logit-RGB space. A zero-initialized output
layer makes the initial mapping close to identity; paired supervision learns
the required exposure, color and detail corrections from low/high image pairs.
"""
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activate: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.SiLU(inplace=False) if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, activate=True)
        self.conv2 = DepthwiseSeparableConv(channels, channels, activate=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + 0.2 * self.conv2(self.conv1(x)))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=False),
        )
        self.refine = nn.Sequential(ResidualBlock(out_channels), ResidualBlock(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=False),
        )
        self.refine = nn.Sequential(ResidualBlock(out_channels), ResidualBlock(out_channels))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.fuse(torch.cat([x, skip], dim=1)))


class PairedRGBRestorationNet(nn.Module):
    """A compact, multi-scale U-Net for supervised low-light RGB restoration."""

    def __init__(self, base_channels: int = 24):
        super().__init__()
        if base_channels < 8:
            raise ValueError("base_channels must be >= 8.")

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 6
        self.base_channels = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(c1), c1),
            nn.SiLU(inplace=False),
            ResidualBlock(c1),
        )
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.bottleneck = DownBlock(c3, c4)

        self.dec3 = UpBlock(c4, c3, c3)
        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)

        self.rgb_residual = nn.Conv2d(c1, 3, kernel_size=3, padding=1)
        # Start at an identity-like mapping, then learn RGB corrections from pairs.
        nn.init.zeros_(self.rgb_residual.weight)
        nn.init.zeros_(self.rgb_residual.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input (B, 3, H, W); got {tuple(x.shape)}.")
        if not torch.is_floating_point(x):
            raise TypeError("Input must be a floating-point RGB tensor in [0, 1].")

        original_h, original_w = x.shape[-2:]
        pad_h = (8 - original_h % 8) % 8
        pad_w = (8 - original_w % 8) % 8
        if pad_h or pad_w:
            pad_mode = "reflect" if original_h > pad_h and original_w > pad_w else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

        x1 = self.stem(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.bottleneck(x3)

        y = self.dec3(x4, x3)
        y = self.dec2(y, x2)
        y = self.dec1(y, x1)

        residual = self.rgb_residual(y)
        # Logit-space residual can brighten dark values without a hard residual cap.
        base_logits = torch.logit(x.clamp(1e-4, 1.0 - 1e-4))
        enhanced = torch.sigmoid(base_logits + residual)
        return enhanced[:, :, :original_h, :original_w]

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_paired_rgb_restoration_net(config: Dict[str, Any] | None = None) -> PairedRGBRestorationNet:
    """Build the network from an optional model/config mapping."""
    base_channels = 24
    if config:
        model_cfg = config.get("model", config)
        base_channels = int(model_cfg.get("base_channels", base_channels))
    return PairedRGBRestorationNet(base_channels=base_channels)
