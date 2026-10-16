"""
Module A: Illumination-Noise Estimator (INE)
Predicts:
- L: Illumination map in (0, 1] (or bounded [0, 1])
- N: Noise proxy / uncertainty map in [0, 1]

Scientific note: N represents a learned noise proxy or uncertainty estimate
unless calibrated against ground-truth noise measurements.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-Separable 2D Convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class IlluminationNoiseEstimator(nn.Module):
    """
    Lightweight Encoder-Decoder using Depthwise-Separable convolutions.
    Inputs:
        I_low: (B, 3, H, W) normalized to [0, 1]
    Outputs:
        L: (B, 1, H, W) Illumination map in [eps, 1]
        N: (B, 1, H, W) Noise uncertainty map in [0, 1]
    """

    def __init__(
        self,
        in_channels: int = 4,  # 3 RGB + 1 Luminance Y
        base_channels: int = 24,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.eps = eps

        # Encoder Stage 1 (Resolution: H, W)
        self.enc1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # Encoder Stage 2 (Resolution: H/2, W/2)
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.enc2 = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # Bottleneck (Resolution: H/4, W/4)
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # Decoder Stage 1 (Resolution: H/2, W/2)
        self.dec1 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # Decoder Stage 2 (Resolution: H, W)
        self.dec2 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 3, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # Output Heads
        # Illumination head: Sigmoid bounded with small eps floor for numerical stability
        self.head_L = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Noise proxy head: Sigmoid in [0, 1]
        self.head_N = nn.Sequential(
            DepthwiseSeparableConv(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute luminance Y = 0.299R + 0.587G + 0.114B
        Shape: (B, 3, H, W) -> (B, 1, H, W)
        """
        r = rgb[:, 0:1, :, :]
        g = rgb[:, 1:2, :, :]
        b = rgb[:, 2:3, :, :]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input RGB tensor (B, 3, H, W) in [0, 1]
        Returns:
            L: Illumination map (B, 1, H, W)
            N: Noise proxy map (B, 1, H, W)
        """
        b, c, h, w = x.shape
        y = self.rgb_to_luminance(x)
        inp = torch.cat([x, y], dim=1)  # (B, 4, H, W)

        # Encoder
        feat1 = self.enc1(inp)  # (B, C, H, W)
        feat2 = self.enc2(self.down1(feat1))  # (B, 2C, H/2, W/2)
        bottle = self.bottleneck(self.down2(feat2))  # (B, 2C, H/4, W/4)

        # Decoder with Bilinear Upsampling & Skip Connections
        up1 = F.interpolate(bottle, size=feat2.shape[2:], mode="bilinear", align_corners=False)
        cat1 = torch.cat([up1, feat2], dim=1)
        dec1_out = self.dec1(cat1)

        up2 = F.interpolate(dec1_out, size=feat1.shape[2:], mode="bilinear", align_corners=False)
        cat2 = torch.cat([up2, feat1], dim=1)
        dec2_out = self.dec2(cat2)

        # Restore exact original resolution if any odd rounding occurred
        if dec2_out.shape[2:] != (h, w):
            dec2_out = F.interpolate(dec2_out, size=(h, w), mode="bilinear", align_corners=False)

        l_map = self.head_L(dec2_out)
        l_map = torch.clamp(l_map, min=self.eps, max=1.0)

        n_map = self.head_N(dec2_out)

        return l_map, n_map
