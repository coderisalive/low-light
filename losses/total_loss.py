"""
Total Self-Supervised Composite Loss
Combines all independently configurable loss terms:
    L_total = λ_mask * L_mask +
              λ_exp * L_exp +
              λ_smooth * L_smooth +
              λ_color * L_color +
              λ_struct * L_struct +
              λ_identity * L_identity
Includes numerical safety checks for NaN and Inf.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

from losses.masked_consistency import MaskedBlindSpotConsistencyLoss
from losses.exposure_loss import PatchExposureControlLoss
from losses.illumination_smoothness import EdgeAwareIlluminationSmoothnessLoss
from losses.color_consistency import ColorConsistencyLoss
from losses.structural_loss import StructuralConsistencyLoss
from losses.identity_regularization import IdentityRegularizationLoss


class LumaGuardTotalLoss(nn.Module):
    """
    Composite loss manager for self-supervised LumaGuard-Net optimization.
    Configurable weights and ablation switches for all candidate loss terms.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__()
        loss_cfg = config.get("loss", {}) if config else {}

        # Loss weights
        self.w_mask = float(loss_cfg.get("w_mask", 0.5))
        self.w_exp = float(loss_cfg.get("w_exp", 8.0))
        self.w_smooth = float(loss_cfg.get("w_smooth", 15.0))
        self.w_color = float(loss_cfg.get("w_color", 3.0))
        self.w_struct = float(loss_cfg.get("w_struct", 0.1))
        self.w_identity = float(loss_cfg.get("w_identity", 0.05))

        # Individual modules
        self.loss_mask = MaskedBlindSpotConsistencyLoss(
            mask_ratio=loss_cfg.get("mask_ratio", 0.15),
            mask_mode=loss_cfg.get("mask_mode", "neighbor_mean"),
        )
        self.loss_exp = PatchExposureControlLoss(
            patch_size=loss_cfg.get("exp_patch_size", 16),
            target_exposure=loss_cfg.get("target_exposure", 0.55),
        )
        self.loss_smooth = EdgeAwareIlluminationSmoothnessLoss(
            k=loss_cfg.get("smooth_k", 10.0)
        )
        self.loss_color = ColorConsistencyLoss(
            patch_size=loss_cfg.get("color_patch_size", 16)
        )
        self.loss_struct = StructuralConsistencyLoss()
        self.loss_identity = IdentityRegularizationLoss()

    def forward(
        self,
        model: nn.Module,
        input_low: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Executes self-supervised forward & loss evaluation.
        Args:
            model: LumaGuardNet model
            input_low: Unpaired low-light input (B, 3, H, W) in [0, 1]
        Returns:
            total_loss: Differentiable scalar loss
            loss_dict: Dictionary of unweighted and weighted loss components
        """
        # 1. Forward pass on original low-light input
        diag = model.forward_diagnostic(input_low)
        enhanced = diag["enhanced"]
        illumination = diag["illumination"]
        q_map = diag["reliability_map"]
        delta_y = diag["delta_y"]
        delta_c = diag["delta_c"]

        # 2. Exposure control loss
        l_exp = self.loss_exp(enhanced) if self.w_exp > 0 else torch.tensor(0.0, device=input_low.device)

        # 3. Edge-aware illumination smoothness loss
        l_smooth = self.loss_smooth(illumination, input_low) if self.w_smooth > 0 else torch.tensor(0.0, device=input_low.device)

        # 4. Color consistency loss
        l_color = self.loss_color(enhanced, input_low, reliability_mask=q_map) if self.w_color > 0 else torch.tensor(0.0, device=input_low.device)

        # 5. Structural consistency loss
        l_struct = self.loss_struct(enhanced, input_low) if self.w_struct > 0 else torch.tensor(0.0, device=input_low.device)

        # 6. Identity / Bounded-correction regularization
        l_identity = self.loss_identity(delta_y, delta_c) if self.w_identity > 0 else torch.tensor(0.0, device=input_low.device)

        # 7. Masked Blind-Spot Consistency (requires secondary masked forward pass)
        if self.w_mask > 0:
            x_masked, mask = self.loss_mask.generate_mask(input_low)
            diag_masked = model.forward_diagnostic(x_masked)
            # Evaluate consistency at held-out positions against original input
            l_mask = self.loss_mask(diag_masked["enhanced"], input_low, mask)
        else:
            l_mask = torch.tensor(0.0, device=input_low.device)

        # Composite total loss
        total_loss = (
            self.w_mask * l_mask
            + self.w_exp * l_exp
            + self.w_smooth * l_smooth
            + self.w_color * l_color
            + self.w_struct * l_struct
            + self.w_identity * l_identity
        )

        # Check for numerical stability (NaN or Inf)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise FloatingPointError(f"Numerical instability detected: total_loss is {total_loss.item()}")

        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_mask": l_mask.item(),
            "loss_exp": l_exp.item(),
            "loss_smooth": l_smooth.item(),
            "loss_color": l_color.item(),
            "loss_struct": l_struct.item(),
            "loss_identity": l_identity.item(),
        }

        return total_loss, loss_dict
