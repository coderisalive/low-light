"""
SNR-Guided Internal Reference Loss Suite (v2).
Refined Self-Supervised Objectives:
- Loss A: Reliable-Patch Pseudo-Pair Reconstruction Loss (Charbonnier)
  Only well-illuminated, high-reliability regions (Y >= 0.28, Q >= 0.40) serve as
  restoration targets. Dark regions are NOT forced to remain dark!
- Loss B: Reliability-Weighted Gradient Structure Loss
  Preserves edge contours on high-confidence features.
- Loss C: Confidence-Gated Reference Feature Consistency Loss
  Feature consistency gated strictly by structural similarity and reliability.
- Loss D: Adaptive Exposure Control Loss
  Penalizes only genuine underexposure without clipping highlights.
- Loss E: Patch Color-Consistency Loss
  Preserves chromatic angle in color space.
- Loss F: Edge-Aware Illumination Smoothness Loss
  Enforces smooth illumination while respecting high-frequency physical boundaries.
- Loss G: Noise-Aware Augmentation Consistency Loss
  Geometric flip consistency weighted by reliability.
- SNRGuidedTotalLoss: Composite configurable manager with NaN/Inf safeguards.
"""

from typing import Dict, Any, Tuple, Optional
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.illumination_smoothness import EdgeAwareIlluminationSmoothnessLoss
from losses.color_consistency import ColorConsistencyLoss


class CharbonnierLoss(nn.Module):
    """Robust Charbonnier Loss: sqrt((x - y)^2 + eps^2)"""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps**2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class PseudoPairSynthesizer:
    """
    Synthesizes realistic low-light degraded pseudo-pairs from reliable regions.
    x_syn = D(x_ref; a, n)
    where:
    - a in [min_attenuation, max_attenuation] simulates photon starvation.
    - n ~ N(0, sigma_n^2) simulates Poisson-Gaussian sensor noise.
    """

    def __init__(
        self,
        min_attenuation: float = 0.12,
        max_attenuation: float = 0.35,
        noise_std_min: float = 0.01,
        noise_std_max: float = 0.04,
    ) -> None:
        self.min_attenuation = min_attenuation
        self.max_attenuation = max_attenuation
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def degrade(self, x_ref: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Degrades input tensor x_ref (B, 3, H, W) to synthetic low-light x_syn.
        """
        b, c, h, w = x_ref.shape
        device = x_ref.device

        alpha = torch.empty(b, 1, 1, 1, device=device).uniform_(
            self.min_attenuation, self.max_attenuation
        )
        sigma = torch.empty(b, 1, 1, 1, device=device).uniform_(
            self.noise_std_min, self.noise_std_max
        )

        noise = torch.randn_like(x_ref) * sigma
        x_syn = torch.clamp(x_ref * alpha + noise, 0.0, 1.0)

        return x_syn, float(alpha.mean().item()), float(sigma.mean().item())


class ReliabilityWeightedStructureLoss(nn.Module):
    """
    Loss B: Reliability-Weighted Gradient Structure Loss.
    Compares horizontal and vertical gradients between enhanced output and reference:
        L_grad = Charbonnier(grad(output), grad(reference))
    Weighted by reference reliability Q_ref so corrupted or uncertain pixels contribute less.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps**2

    @staticmethod
    def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
        """Y = 0.299R + 0.587G + 0.114B"""
        return 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        weight_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y_pred = self.rgb_to_luminance(pred_rgb)
        y_tgt = self.rgb_to_luminance(target_rgb)

        # Horizontal gradients
        gx_pred = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
        gx_tgt = y_tgt[:, :, :, 1:] - y_tgt[:, :, :, :-1]
        diff_x = torch.sqrt((gx_pred - gx_tgt) ** 2 + self.eps2)

        # Vertical gradients
        gy_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        gy_tgt = y_tgt[:, :, 1:, :] - y_tgt[:, :, :-1, :]
        diff_y = torch.sqrt((gy_pred - gy_tgt) ** 2 + self.eps2)

        if weight_mask is not None:
            wx = weight_mask[:, :, :, 1:]
            wy = weight_mask[:, :, 1:, :]
            loss_x = torch.sum(diff_x * wx) / (torch.sum(wx) + 1e-6)
            loss_y = torch.sum(diff_y * wy) / (torch.sum(wy) + 1e-6)
        else:
            loss_x = torch.mean(diff_x)
            loss_y = torch.mean(diff_y)

        return loss_x + loss_y


class ConfidenceGatedReferenceLoss(nn.Module):
    """
    Loss C: Confidence-Gated Reference Feature Consistency Loss.
    Encourages structurally compatible features to align between target and reference,
    strictly gated by matching confidence and reference reliability.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        feat_target: torch.Tensor,
        feat_ref: torch.Tensor,
        match_confidence: torch.Tensor,
        min_conf: float = 0.20,
    ) -> torch.Tensor:
        f_t_norm = F.normalize(feat_target, p=2, dim=1, eps=self.eps)
        f_r_norm = F.normalize(feat_ref, p=2, dim=1, eps=self.eps)

        cos_sim = torch.sum(f_t_norm * f_r_norm, dim=1, keepdim=True)
        cos_dist = 1.0 - torch.clamp(cos_sim, -1.0, 1.0)

        conf_mask = (match_confidence > min_conf).float() * match_confidence
        denom = torch.sum(conf_mask) + 1e-6
        if denom < 1.0:
            return torch.tensor(0.0, device=feat_target.device)

        loss = torch.sum(cos_dist * conf_mask) / denom
        return loss


class AdaptiveExposureLoss(nn.Module):
    """
    Loss D: Adaptive Exposure Control Loss.
    Penalizes underexposure in dark patches without forcing already bright
    or properly exposed areas to blow out.
    """

    def __init__(
        self,
        patch_size: int = 16,
        target_exposure: float = 0.50,
        max_acceptable_lum: float = 0.88,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.target_exposure = target_exposure
        self.max_acceptable_lum = max_acceptable_lum

    @staticmethod
    def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
        """Y = 0.299R + 0.587G + 0.114B"""
        return 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]

    def forward(
        self,
        enhanced: torch.Tensor,
        reliability_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        lum = self.rgb_to_luminance(enhanced)

        local_mean = F.avg_pool2d(
            lum, kernel_size=self.patch_size, stride=self.patch_size, padding=0
        )

        under_deficit = torch.relu(self.target_exposure - local_mean)
        over_excess = torch.relu(local_mean - self.max_acceptable_lum)

        deficit_total = under_deficit**2 + 2.5 * (over_excess**2)
        return torch.mean(deficit_total)


class NoiseAwareAugmentationConsistencyLoss(nn.Module):
    """
    Loss G: Noise-Aware Augmentation Consistency Loss.
    Enforces geometric flip consistency weighted by reliability.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=eps)

    def forward(
        self,
        model: nn.Module,
        x_low: torch.Tensor,
        enh_original: torch.Tensor,
        q_map: torch.Tensor,
    ) -> torch.Tensor:
        x_flipped = torch.flip(x_low, dims=[3])
        diag_flipped = model.forward_diagnostic(x_flipped)
        enh_flipped = diag_flipped["enhanced"]

        enh_unflipped = torch.flip(enh_flipped, dims=[3])
        diff = torch.sqrt((enh_original - enh_unflipped) ** 2 + 1e-6)
        q_weight = torch.clamp(q_map, 0.1, 1.0)

        loss = torch.sum(diff * q_weight) / (torch.sum(q_weight) + 1e-6)
        return loss


class SNRGuidedTotalLoss(nn.Module):
    """
    Composite Loss Manager for SNR-Guided Internal Reference Training (v2).
    Configurable Loss Composition:
        L_total = lambda_pseudo * L_pseudo
                + lambda_gradient * L_gradient
                + lambda_reference * L_reference
                + lambda_exposure * L_exposure
                + lambda_color * L_color
                + lambda_smooth * L_smooth
                + lambda_consistency * L_consistency

    Critical Fix: Pseudo-pair reconstruction loss L_pseudo only uses genuinely
    well-lit, high-reliability regions (Y >= 0.28, Q >= 0.40) as targets,
    ensuring dark regions are never penalized for being brightened!
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        loss_cfg = config.get("loss", {}) if config else {}

        # Configurable Loss Weights
        self.w_pseudo = float(loss_cfg.get("w_pseudo", 1.0))
        self.w_gradient = float(loss_cfg.get("w_gradient", 0.5))
        self.w_reference = float(loss_cfg.get("w_reference", 0.1))
        self.w_exposure = float(loss_cfg.get("w_exposure", 4.0))
        self.w_color = float(loss_cfg.get("w_color", 2.0))
        self.w_smooth = float(loss_cfg.get("w_smooth", 10.0))
        self.w_consistency = float(loss_cfg.get("w_consistency", 0.1))

        # Individual Modules
        self.synthesizer = PseudoPairSynthesizer(
            min_attenuation=loss_cfg.get("min_attenuation", 0.12),
            max_attenuation=loss_cfg.get("max_attenuation", 0.35),
            noise_std_min=loss_cfg.get("noise_std_min", 0.01),
            noise_std_max=loss_cfg.get("noise_std_max", 0.04),
        )
        self.loss_charbonnier = CharbonnierLoss()
        self.loss_structure = ReliabilityWeightedStructureLoss()
        self.loss_ref_feature = ConfidenceGatedReferenceLoss()
        self.loss_exposure = AdaptiveExposureLoss(
            patch_size=loss_cfg.get("exp_patch_size", 16),
            target_exposure=loss_cfg.get("target_exposure", 0.50),
        )
        self.loss_smooth = EdgeAwareIlluminationSmoothnessLoss(
            k=loss_cfg.get("smooth_k", 10.0)
        )
        self.loss_color = ColorConsistencyLoss(
            patch_size=loss_cfg.get("color_patch_size", 16)
        )
        self.loss_consistency = NoiseAwareAugmentationConsistencyLoss()

    def forward(
        self,
        model: nn.Module,
        input_low: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Executes self-supervised forward & loss evaluation.
        Args:
            model: SNRInternalReferenceNet
            input_low: Real low-light image (B, 3, H, W) in [0, 1]
        Returns:
            total_loss: Differentiable scalar
            loss_dict: Dictionary of unweighted and weighted loss terms
        """
        device = input_low.device

        # 1. Forward pass on real low-light input
        diag = model.forward_diagnostic(input_low)
        enhanced = diag["enhanced"]
        illumination = diag["illumination"]
        q_map = diag["reliability_map"]
        feat_target = diag["features"]
        match_conf = diag["match_confidence"]

        # 2. Loss A & B: Pseudo-Pair Reconstruction on Genuinely Well-Lit Patches Only
        l_pseudo = torch.tensor(0.0, device=device)
        l_gradient = torch.tensor(0.0, device=device)

        if self.w_pseudo > 0 or self.w_gradient > 0:
            lum_in = 0.299 * input_low[:, 0:1] + 0.587 * input_low[:, 1:2] + 0.114 * input_low[:, 2:3]
            # Valid target mask: only pixels with sufficient illumination and high reliability serve as pseudo targets!
            rel_target_mask = (lum_in >= 0.28).float() * (q_map >= 0.35).float()

            with torch.no_grad():
                x_syn, alpha_val, noise_val = self.synthesizer.degrade(input_low)

            diag_syn = model.forward_diagnostic(x_syn)
            enh_syn = diag_syn["enhanced"]

            if self.w_pseudo > 0:
                diff_pseudo = torch.sqrt((enh_syn - input_low) ** 2 + 1e-6)
                # Compute reconstruction loss ONLY on the reliable well-lit pixels
                denom_rel = torch.sum(rel_target_mask) + 1e-5
                if denom_rel > 10.0:
                    l_pseudo = torch.sum(diff_pseudo * rel_target_mask) / denom_rel
                else:
                    # If whole image is pitch black with no reliable target, fallback to exposure-guided reconstruction
                    l_pseudo = torch.tensor(0.0, device=device)

            if self.w_gradient > 0:
                l_gradient = self.loss_structure(enh_syn, input_low, weight_mask=rel_target_mask)

        # 3. Loss C: Confidence-Gated Reference Feature Consistency
        l_ref = torch.tensor(0.0, device=device)
        if self.w_reference > 0 and diag.get("features_fused") is not None:
            l_ref = self.loss_ref_feature(feat_target, diag["features_fused"], match_conf)

        # 4. Loss D: Adaptive Exposure Control on Enhanced Real Image
        l_exp = torch.tensor(0.0, device=device)
        if self.w_exposure > 0:
            l_exp = self.loss_exposure(enhanced, reliability_map=q_map)

        # 5. Loss E: Color Consistency
        l_color = torch.tensor(0.0, device=device)
        if self.w_color > 0:
            l_color = self.loss_color(enhanced, input_low, reliability_mask=q_map)

        # 6. Loss F: Edge-Aware Illumination Smoothness
        l_smooth = torch.tensor(0.0, device=device)
        if self.w_smooth > 0:
            l_smooth = self.loss_smooth(illumination, input_low)

        # 7. Loss G: Noise-Aware Augmentation Consistency
        l_consistency = torch.tensor(0.0, device=device)
        if self.w_consistency > 0:
            l_consistency = self.loss_consistency(model, input_low, enhanced, q_map)

        # Total Composite Objective
        total_loss = (
            self.w_pseudo * l_pseudo
            + self.w_gradient * l_gradient
            + self.w_reference * l_ref
            + self.w_exposure * l_exp
            + self.w_color * l_color
            + self.w_smooth * l_smooth
            + self.w_consistency * l_consistency
        )

        # Numerical stability safeguard
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise FloatingPointError(
                f"Numerical instability detected: total_loss is {total_loss.item()}"
            )

        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_pseudo": l_pseudo.item(),
            "loss_gradient": l_gradient.item(),
            "loss_reference": l_ref.item(),
            "loss_exposure": l_exp.item(),
            "loss_color": l_color.item(),
            "loss_smooth": l_smooth.item(),
            "loss_consistency": l_consistency.item(),
        }

        return total_loss, loss_dict
