"""
Loss A: Masked Blind-Spot Consistency Loss
L_mask = mean(abs(mask * (prediction - observed_target)))

Scientific Documentation:
This objective creates stochastic views of low-light images using a structured
blind-spot pixel mask. In masked positions, the target pixel is held out and
interpolated from context, forcing the network to model spatial coherence rather
than learning a trivial identity copy.

Note: As required by research principles, this objective learns statistics of
observed low-light images and does NOT independently define the correct target brightness.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBlindSpotConsistencyLoss(nn.Module):
    """
    Masked Blind-Spot Consistency Loss.
    Masks a fraction of input pixels using neighbor-averaging or dropout perturbation,
    and computes consistency between the network's prediction on the masked input
    and the clean observed pixels at the held-out locations.
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_mode: str = "neighbor_mean",  # 'neighbor_mean', 'zero', or 'noise'
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode

    def generate_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate binary mask M and perturbed input x_masked.
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            x_masked: Perturbed tensor
            mask: Binary mask (B, 1, H, W) where 1 indicates masked pixels
        """
        b, c, h, w = x.shape
        # Sample Bernoulli mask
        rand_tensor = torch.rand((b, 1, h, w), device=x.device, dtype=x.dtype)
        mask = (rand_tensor < self.mask_ratio).float()

        if self.mask_mode == "neighbor_mean":
            # 3x3 uniform filter excluding center
            kernel = torch.tensor(
                [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                device=x.device,
                dtype=x.dtype,
            ).view(1, 1, 3, 3) / 8.0
            kernel = kernel.repeat(c, 1, 1, 1)

            # Replicate channels for depthwise convolution
            neighbor_mean = F.conv2d(x, kernel, padding=1, groups=c)
            x_masked = (1.0 - mask) * x + mask * neighbor_mean
        elif self.mask_mode == "noise":
            noise = torch.randn_like(x) * 0.1
            x_masked = (1.0 - mask) * x + mask * torch.clamp(x + noise, 0.0, 1.0)
        else:
            x_masked = (1.0 - mask) * x

        return x_masked, mask

    def forward(
        self,
        prediction: torch.Tensor,
        observed_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates consistency: mean(abs(mask * (prediction - observed_target)))
        Args:
            prediction: Model output on perturbed/masked input (B, C, H, W)
            observed_target: Unperturbed observed input (B, C, H, W)
            mask: Binary mask where 1 indicates masked positions (B, 1, H, W)
        """
        diff = torch.abs(prediction - observed_target)
        mask_expanded = mask.expand_as(diff)
        denom = torch.sum(mask_expanded) + 1e-6
        loss = torch.sum(mask_expanded * diff) / denom
        return loss
