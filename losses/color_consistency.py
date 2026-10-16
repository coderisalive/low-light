"""
Loss D: Color Consistency Loss
Discourages unnatural chromaticity drifts by penalizing color vector angle differences
or inter-channel deviations between original and enhanced images, selectively masked
in reliable regions to avoid copying severe low-light color noise.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorConsistencyLoss(nn.Module):
    """
    Color Consistency Loss.
    Uses cosine similarity or patch-averaged color deviations between enhanced and input image.
    Optionally modulated by the reliability map Q to only enforce color constancy
    in high-confidence regions.
    """

    def __init__(self, patch_size: int = 16, eps: float = 1e-6) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps

    def forward(
        self,
        enhanced: torch.Tensor,
        original: torch.Tensor,
        reliability_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            enhanced: Enhanced RGB tensor (B, 3, H, W)
            original: Input low-light RGB tensor (B, 3, H, W)
            reliability_mask: Optional reliability weighting (B, 1, H, W)
        """
        # Patch-average colors
        p_enh = F.avg_pool2d(
            enhanced, kernel_size=self.patch_size, stride=self.patch_size, padding=0
        )
        p_orig = F.avg_pool2d(
            original, kernel_size=self.patch_size, stride=self.patch_size, padding=0
        )

        # Compute cosine angle deviation in color space across patches
        norm_enh = torch.norm(p_enh, p=2, dim=1, keepdim=True) + self.eps
        norm_orig = torch.norm(p_orig, p=2, dim=1, keepdim=True) + self.eps

        cos_sim = torch.sum((p_enh / norm_enh) * (p_orig / norm_orig), dim=1, keepdim=True)
        # Cosine distance: 1 - cos_sim in [0, 2]
        color_dist = 1.0 - torch.clamp(cos_sim, -1.0, 1.0)

        if reliability_mask is not None:
            p_rel = F.avg_pool2d(
                reliability_mask, kernel_size=self.patch_size, stride=self.patch_size, padding=0
            )
            return torch.sum(color_dist * p_rel) / (torch.sum(p_rel) + self.eps)

        return torch.mean(color_dist)
