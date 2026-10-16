"""
Loss B: Patch-Based Exposure Control Loss
L_exp = mean(abs(local_mean(Y_out) - target_exposure))

Uses a configurable patch size (e.g. 16x16) and target exposure (e.g. 0.55-0.60).
To prevent overexposing naturally dark scene content, the loss can be weighted
by an illumination-aware or edge-aware weighting map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchExposureControlLoss(nn.Module):
    """
    Patch-based Exposure Control Loss computed over local spatial windows.
    Prevents both under-exposure and systemic over-saturation.
    """

    def __init__(
        self,
        patch_size: int = 16,
        target_exposure: float = 0.55,
        use_l1: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.target_exposure = target_exposure
        self.use_l1 = use_l1

    @staticmethod
    def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
        """Y = 0.299R + 0.587G + 0.114B"""
        return 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]

    def forward(
        self,
        enhanced: torch.Tensor,
        weight_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            enhanced: Enhanced RGB tensor (B, 3, H, W)
            weight_mask: Optional spatial importance weighting (B, 1, H, W)
        """
        # Compute luminance
        lum = self.rgb_to_luminance(enhanced)

        # Average pool across local patches
        local_mean = F.avg_pool2d(
            lum,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            ceil_mode=False,
        )

        target = torch.full_like(local_mean, self.target_exposure)

        if self.use_l1:
            diff = torch.abs(local_mean - target)
        else:
            diff = torch.pow(local_mean - target, 2)

        if weight_mask is not None:
            # Pool weight mask to match patch resolution
            w_pooled = F.avg_pool2d(
                weight_mask,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                padding=0,
                ceil_mode=False,
            )
            return torch.sum(diff * w_pooled) / (torch.sum(w_pooled) + 1e-6)

        return torch.mean(diff)
