"""
Loss E: Structural Consistency Loss
Measures structural similarity between input high-frequency content and enhanced output.
Uses a differentiable SSIM-based or gradient-correlation formulation.
Avoids forcing raw intensity equality to prevent under-enhancement.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Creates a 2D Gaussian kernel."""
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.repeat(channels, 1, 1, 1)
    return window


class StructuralConsistencyLoss(nn.Module):
    """
    Differentiable Structural Consistency Loss (1 - SSIM) on luminance or RGB.
    Preserves contours and structures without penalizing illumination changes.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", create_gaussian_window(window_size, sigma, channels))

    @staticmethod
    def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
        """Y = 0.299R + 0.587G + 0.114B"""
        return 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]

    def ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size(1)
        if self.window.size(0) != channel or self.window.device != img1.device:
            self.window = create_gaussian_window(self.window_size, 1.5, channel).to(img1.device)

        c1 = 0.01**2
        c2 = 0.03**2

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel)
            - mu1_mu2
        )

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return ssim_map.mean()

    def forward(self, enhanced: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        Calculates 1 - SSIM on luminance channels.
        """
        y_enh = self.rgb_to_luminance(enhanced)
        y_orig = self.rgb_to_luminance(original)
        return 1.0 - self.ssim(y_enh, y_orig)
