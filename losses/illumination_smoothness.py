"""
Loss C: Edge-Aware Illumination Smoothness Loss
L_smooth = mean(
    abs(grad_x(L)) * exp(-k * abs(grad_x(Y))) +
    abs(grad_y(L)) * exp(-k * abs(grad_y(Y)))
)
Enforces spatial smoothness on the illumination map L, while preserving boundaries
corresponding to high-frequency image edges in Y.
"""

import torch
import torch.nn as nn


class EdgeAwareIlluminationSmoothnessLoss(nn.Module):
    """
    Edge-aware Illumination Smoothness Loss.
    Ensures illumination transitions are piecewise smooth, respecting true scene boundaries.
    """

    def __init__(self, k: float = 10.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.k = k
        self.eps = eps

    @staticmethod
    def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
        """Y = 0.299R + 0.587G + 0.114B"""
        return 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]

    def forward(self, illumination: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            illumination: Predicted illumination map L (B, 1, H, W)
            input_image: Input RGB image (B, 3, H, W)
        """
        lum = self.rgb_to_luminance(input_image)

        # Gradients of illumination map L
        l_grad_x = torch.abs(illumination[:, :, :, 1:] - illumination[:, :, :, :-1])
        l_grad_y = torch.abs(illumination[:, :, 1:, :] - illumination[:, :, :-1, :])

        # Gradients of luminance Y
        y_grad_x = torch.abs(lum[:, :, :, 1:] - lum[:, :, :, :-1])
        y_grad_y = torch.abs(lum[:, :, 1:, :] - lum[:, :, :-1, :])

        # Edge-attenuated weights
        weight_x = torch.exp(-self.k * y_grad_x)
        weight_y = torch.exp(-self.k * y_grad_y)

        loss_x = torch.mean(l_grad_x * weight_x)
        loss_y = torch.mean(l_grad_y * weight_y)

        return loss_x + loss_y
