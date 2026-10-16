"""
Module D: Bounded Residual Enhancement Head
Dual output heads:
- Luminance correction head (predicts ΔY)
- Chrominance correction head (predicts ΔC)
Bounded residual equations:
    Y_out = clamp(Y + alpha * ΔY, 0, 1)
    C_out = C + beta * ΔC
Invertible conversion back to RGB with clamp [0, 1].

Also implements an alternative RGB direct residual output head behind a config flag
for ablation comparison.
"""

from typing import Tuple
import torch
import torch.nn as nn
from models.illumination_noise_estimator import DepthwiseSeparableConv


class ColorSpaceConverter:
    """
    Differentiable RGB <-> YCbCr (BT.601 standard) conversion.
    Input/output ranges are normalized to [0, 1] or appropriate ranges.
    """

    @staticmethod
    def rgb_to_ycbcr(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts RGB in [0, 1] to Luminance Y and Chrominance (Cb, Cr).
        Returns:
            Y: (B, 1, H, W) in [0, 1]
            CbCr: (B, 2, H, W)
        """
        r = rgb[:, 0:1, :, :]
        g = rgb[:, 1:2, :, :]
        b = rgb[:, 2:3, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        cbcr = torch.cat([cb, cr], dim=1)
        return y, cbcr

    @staticmethod
    def ycbcr_to_rgb(y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        """
        Converts Y (B, 1, H, W) and CbCr (B, 2, H, W) back to RGB in [0, 1].
        """
        cb = cbcr[:, 0:1, :, :]
        cr = cbcr[:, 1:2, :, :]

        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb

        rgb = torch.cat([r, g, b], dim=1)
        return torch.clamp(rgb, 0.0, 1.0)


class BoundedResidualEnhancementHead(nn.Module):
    """
    Enhancement head predicting bounded residual updates.
    """

    def __init__(
        self,
        feature_channels: int = 32,
        use_separate_color: bool = True,
        init_alpha: float = 0.2,
        init_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_separate_color = use_separate_color

        # Learnable scaling coefficients with positive constraint or small initialization
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))

        if use_separate_color:
            # Luminance head: predicts ΔY in [-1, 1]
            self.head_lum = nn.Sequential(
                DepthwiseSeparableConv(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(feature_channels // 2, 1, kernel_size=1),
                nn.Tanh(),
            )
            # Chrominance head: predicts ΔCbCr in [-1, 1]
            self.head_chrom = nn.Sequential(
                DepthwiseSeparableConv(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(feature_channels // 2, 2, kernel_size=1),
                nn.Tanh(),
            )
            self.head_rgb = None
        else:
            # RGB residual head: predicts ΔRGB directly in [-1, 1]
            self.head_lum = None
            self.head_chrom = None
            self.head_rgb = nn.Sequential(
                DepthwiseSeparableConv(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(feature_channels // 2, 3, kernel_size=1),
                nn.Tanh(),
            )

    def forward(
        self, feat: torch.Tensor, original_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: Guided feature map (B, C, H, W)
            original_rgb: Input RGB image (B, 3, H, W) in [0, 1]
        Returns:
            enhanced_rgb: Enhanced image (B, 3, H, W) in [0, 1]
            delta_y: Luminance residual (B, 1, H, W) or zeros if RGB mode
            delta_c: Chrominance residual (B, 2, H, W) or zeros if RGB mode
        """
        if self.use_separate_color:
            y_in, cbcr_in = ColorSpaceConverter.rgb_to_ycbcr(original_rgb)

            delta_y = self.head_lum(feat)
            delta_c = self.head_chrom(feat)

            # Bounded residual correction
            y_out = torch.clamp(y_in + torch.abs(self.alpha) * delta_y, 0.0, 1.0)
            cbcr_out = cbcr_in + self.beta * delta_c

            enhanced_rgb = ColorSpaceConverter.ycbcr_to_rgb(y_out, cbcr_out)
            return enhanced_rgb, delta_y, delta_c
        else:
            delta_rgb = self.head_rgb(feat)
            enhanced_rgb = torch.clamp(original_rgb + torch.abs(self.alpha) * delta_rgb, 0.0, 1.0)
            zeros_y = torch.zeros((feat.size(0), 1, feat.size(2), feat.size(3)), device=feat.device)
            zeros_c = torch.zeros((feat.size(0), 2, feat.size(2), feat.size(3)), device=feat.device)
            return enhanced_rgb, zeros_y, zeros_c
