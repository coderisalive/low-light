"""
Loss F: Identity & Bounded-Correction Regularization Loss
Regularizes magnitude of residual corrections to avoid hallucinations
and over-amplification in regions that are already sufficiently illuminated.
"""

import torch
import torch.nn as nn


class IdentityRegularizationLoss(nn.Module):
    """
    Penalizes extreme residual magnitude:
    L_reg = mean(abs(delta_y)) + mean(abs(delta_c))
    or L2 on difference between input and enhanced.
    """

    def __init__(self, use_l1: bool = True) -> None:
        super().__init__()
        self.use_l1 = use_l1

    def forward(
        self,
        delta_y: torch.Tensor,
        delta_c: torch.Tensor,
        enhanced: torch.Tensor = None,
        original: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.use_l1:
            loss = torch.mean(torch.abs(delta_y)) + torch.mean(torch.abs(delta_c))
        else:
            loss = torch.mean(delta_y**2) + torch.mean(delta_c**2)
        return loss
