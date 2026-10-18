"""
SNR-Guided Internal Reference Transfer Network (SNRInternalReferenceNet)
for Lightweight Self-Supervised Low-Light Image Enhancement.

Refined Architecture (v2):
- Phase 3 Compliant: Confidence-controlled residual feature fusion:
      F_fused = F_target + alpha * confidence * (F_reference - F_target)
  Prevents ghosting, blotches, and arbitrary feature drift.
- Phase 4 Compliant: Structural saliency noise decoupling:
  Separates coherent structural edges from high-frequency sensor noise.
  Prevents real contours from being penalized as noise.
- Phase 5 Compliant: Chromaticity-preserving Retinex output head:
  Scales chrominance proportionally with luminance gain, eliminating
  muted colors, grey haze, and highlight clipping.
"""

from typing import Dict, Any, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.illumination_noise_estimator import DepthwiseSeparableConv
from models.lightweight_blocks import LightweightResBlock
from models.enhancement_head import ColorSpaceConverter


class ReliabilityEstimator(nn.Module):
    """
    Module A: Refined Reliability & SNR-Proxy Estimator.
    Estimates a continuous reliability map Q in [0, 1] based on:
    - Local signal intensity S (illumination proxy)
    - Directional gradient magnitude (structural edge coherence)
    - Laplacian high-frequency response decoupled from true edges (noise proxy)
    - Saturation penalty P_sat (rejects clipped highlights)
    - Lightweight learnable refinement

    Scientific note: Without calibrated sensor profiles, this acts as a
    principled reliability/SNR proxy rather than a physical decibel measurement.
    """

    def __init__(
        self,
        base_channels: int = 16,
        kernel_size: int = 5,
        saturation_threshold: float = 0.92,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.saturation_threshold = saturation_threshold
        self.eps = eps

        # Learnable scaling coefficients
        self.w_signal = nn.Parameter(torch.tensor(1.2, dtype=torch.float32))
        self.w_struct = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))
        self.w_noise = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

        # Compact convolutional refinement (5 input channels: lum, signal, noise, sat_pen, struct)
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(5, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Fixed Laplacian kernel for isotropic high-frequency response
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", laplacian)

        # Sobel kernels for directional edge energy
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def compute_edge_and_noise_proxies(self, lum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decouple coherent structural edges from isotropic sensor noise.
        - True edges: strong coherent directional gradients (Sobel x & y).
        - Noise: high Laplacian variance without directional alignment.
        """
        # 1. Directional gradient energy
        gx = F.conv2d(lum, self.sobel_x, padding=1)
        gy = F.conv2d(lum, self.sobel_y, padding=1)
        edge_mag = torch.sqrt(gx**2 + gy**2 + self.eps)
        struct_proxy = F.avg_pool2d(
            edge_mag, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2
        )

        # 2. Laplacian high-frequency response
        high_freq = F.conv2d(lum, self.laplacian_kernel, padding=1)
        abs_hf = torch.abs(high_freq)
        hf_pool = F.avg_pool2d(
            abs_hf, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2
        )

        # 3. Decouple: noise proxy is high-frequency variance MINUS coherent edge energy
        noise_proxy = torch.clamp(hf_pool - 0.6 * struct_proxy, 0.0, 1.0)
        struct_proxy = torch.clamp(struct_proxy * 3.0, 0.0, 1.0)

        return struct_proxy, noise_proxy

    def compute_saturation_penalty(self, lum: torch.Tensor) -> torch.Tensor:
        """
        Smooth saturation penalty: P_sat -> 0 for pixels near or above saturation_threshold.
        Prevents overexposed/clipped regions from acting as structural references.
        """
        steepness = 25.0
        penalty = torch.sigmoid(steepness * (self.saturation_threshold - lum))
        return torch.clamp(penalty, 0.0, 1.0)

    def forward(
        self, x: torch.Tensor, lum: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input RGB tensor (B, 3, H, W) in [0, 1]
            lum: Luminance tensor (B, 1, H, W) in [0, 1]
        Returns:
            q_map: Reliability map (B, 1, H, W) in [0, 1]
            noise_proxy: Estimated noise map (B, 1, H, W)
            sat_penalty: Saturation penalty map (B, 1, H, W)
        """
        # 1. Local signal intensity
        signal = F.avg_pool2d(
            lum, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2
        )

        # 2. Decoupled structural and noise proxies
        struct_proxy, noise_proxy = self.compute_edge_and_noise_proxies(lum)

        # 3. Saturation penalty
        sat_penalty = self.compute_saturation_penalty(lum)

        # 4. Principled proxy formulation:
        # Boosted by signal and coherent structure, penalized by noise
        log_signal = torch.log(torch.clamp(signal, min=self.eps, max=1.0))
        log_signal = torch.clamp(log_signal, min=-10.0, max=0.0)
        logit = (
            self.w_signal * log_signal
            + self.w_struct * struct_proxy
            - self.w_noise * noise_proxy
            + self.bias
        )
        phys_proxy = torch.sigmoid(torch.clamp(logit, min=-12.0, max=12.0))

        # 5. Convolutional refinement for multi-channel spatial context
        inp_feat = torch.cat([lum, signal, noise_proxy, sat_penalty, struct_proxy], dim=1)
        learned_refinement = self.refine(inp_feat)

        # Combined reliability: strictly zeroed at clipped highlights
        q_map = sat_penalty * (0.65 * phys_proxy + 0.35 * learned_refinement)
        q_map = torch.clamp(q_map, 0.0, 1.0)

        return q_map, noise_proxy, sat_penalty


class InternalReferenceMatcher(nn.Module):
    """
    Module C: High-Reliability Internal Reference Extractor & Matcher.
    Extracts structural reference features from high-reliability patches and
    performs structural patch matching using normalized feature similarity.

    Design constraints:
    - Avoids expensive O(N^2) all-to-all pixel attention.
    - Uses grid-pooled patch tokens (e.g. 16x16 pixels per patch).
    - Excludes trivial self-matching (target patch excluded from candidate set).
    - Downweights or rejects low-reliability (< threshold) or saturated references.
    - Selects Top-K structurally similar references with safe softmax normalization.
    - Provides smooth, continuous fallback when no trustworthy reference exists.
    """

    def __init__(
        self,
        feature_channels: int = 32,
        patch_size: int = 16,
        top_k: int = 3,
        min_ref_reliability: float = 0.40,
        similarity_threshold: float = 0.25,
        temperature: float = 0.15,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.patch_size = patch_size
        self.top_k = top_k
        self.min_ref_reliability = min_ref_reliability
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.eps = eps

        # Lightweight linear projections for query, key, value
        self.proj_q = nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=1, bias=False)
        self.proj_k = nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=1, bias=False)
        self.proj_v = nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False)

    def forward(
        self,
        feat: torch.Tensor,
        q_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: Feature map (B, C, H, W)
            q_map: Reliability map (B, 1, H, W) in [0, 1]
        Returns:
            ref_feat_dense: Dense transferred reference feature map (B, C, H, W)
            match_conf_dense: Dense matching confidence map (B, 1, H, W)
            ref_q_dense: Dense reliability of selected references (B, 1, H, W)
        """
        b, c, h, w = feat.shape

        # 1. Project features for matching
        q_proj = self.proj_q(feat)  # (B, C/2, H, W)
        k_proj = self.proj_k(feat)  # (B, C/2, H, W)
        v_proj = self.proj_v(feat)  # (B, C, H, W)

        # 2. Grid-level patch pooling
        h_p = max(1, h // self.patch_size)
        w_p = max(1, w // self.patch_size)

        q_tokens = F.adaptive_avg_pool2d(q_proj, (h_p, w_p)).flatten(2).transpose(1, 2)
        k_tokens = F.adaptive_avg_pool2d(k_proj, (h_p, w_p)).flatten(2).transpose(1, 2)
        v_tokens = F.adaptive_avg_pool2d(v_proj, (h_p, w_p)).flatten(2).transpose(1, 2)
        q_patch = F.adaptive_avg_pool2d(q_map, (h_p, w_p)).flatten(2).squeeze(1)  # (B, M)

        m = q_tokens.size(1)  # total number of patches

        # 3. Structural feature normalization (L2-norm along channel dimension)
        q_norm = F.normalize(q_tokens, p=2, dim=-1, eps=self.eps)  # (B, M, C/2)
        k_norm = F.normalize(k_tokens, p=2, dim=-1, eps=self.eps)  # (B, M, C/2)

        # Pairwise structural cosine similarity: (B, M, M)
        sim = torch.bmm(q_norm, k_norm.transpose(1, 2))  # values in [-1, 1]

        # 4. Validity Masking:
        # Exclude self-matching: diagonal = -inf
        diag_mask = torch.eye(m, device=feat.device, dtype=torch.bool).unsqueeze(0)  # (1, M, M)

        # Exclude low-reliability reference candidates: patch j must have q_patch[j] >= min_ref_reliability
        ref_valid = (q_patch >= self.min_ref_reliability).unsqueeze(1)  # (B, 1, M)

        # Combined validity mask
        valid_mask = (~diag_mask) & ref_valid  # (B, M, M)

        # Keep masked similarities and the temperature softmax in FP32.
        # In FP16, -1e4 / temperature can overflow to -inf; rows with no valid
        # candidates can then produce softmax(NaN) when every score is -inf.
        sim_fp32 = sim.float()
        masked_sim = torch.where(
            valid_mask,
            sim_fp32,
            torch.full_like(sim_fp32, -1e4),
        )

        # 5. Top-K Selection with safe normalization
        k_val = min(self.top_k, max(1, m - 1))
        topk_sim, topk_idx = torch.topk(masked_sim, k=k_val, dim=-1)  # (B, M, K)

        # Check if valid candidates actually exist
        has_valid_candidates = (topk_sim > -100.0).any(dim=-1, keepdim=True)  # (B, M, 1)

        # Temperature-scaled FP32 softmax; explicitly zero rows with no valid refs.
        topk_weights = F.softmax(topk_sim / self.temperature, dim=-1)
        topk_weights = topk_weights * has_valid_candidates.to(topk_weights.dtype)
        topk_weights = topk_weights.to(dtype=v_tokens.dtype)

        # Gather Top-K value tokens and Top-K reliabilities via vectorized indexing
        batch_idx = torch.arange(b, device=feat.device).view(b, 1, 1).expand(-1, m, k_val)
        gathered_v = v_tokens[batch_idx, topk_idx]  # (B, M, K, C)
        gathered_q = q_patch[batch_idx, topk_idx]  # (B, M, K)

        # Weighted combination of Top-K reference features
        ref_tokens = torch.sum(topk_weights.unsqueeze(-1) * gathered_v, dim=2)  # (B, M, C)
        ref_q_tokens = torch.sum(topk_weights * gathered_q, dim=-1, keepdim=True)  # (B, M, 1)

        # 6. Compute matching confidence:
        # Confidence depends on:
        # (a) maximum structural similarity exceeding similarity_threshold
        # (b) reliability of the chosen reference patch
        max_sim = topk_sim[:, :, 0:1]  # (B, M, 1)
        sim_conf = torch.clamp(
            (max_sim - self.similarity_threshold) / (1.0 - self.similarity_threshold + self.eps),
            0.0,
            1.0,
        )
        match_conf = sim_conf * ref_q_tokens * has_valid_candidates.float()

        # Keep reference features unscaled here. Confidence is applied exactly once
        # by ConfidenceGatedFeatureTransfer; scaling here as well caused a double
        # attenuation and made the transferred representation unnecessarily weak.

        # 7. Reshape tokens back to 2D grid and interpolate to full resolution (H, W)
        ref_grid = ref_tokens.transpose(1, 2).view(b, c, h_p, w_p)
        conf_grid = match_conf.transpose(1, 2).view(b, 1, h_p, w_p)
        ref_q_grid = ref_q_tokens.transpose(1, 2).view(b, 1, h_p, w_p)

        ref_feat_dense = F.interpolate(
            ref_grid, size=(h, w), mode="bilinear", align_corners=False
        )
        match_conf_dense = F.interpolate(
            conf_grid, size=(h, w), mode="bilinear", align_corners=False
        )
        ref_q_dense = F.interpolate(
            ref_q_grid, size=(h, w), mode="bilinear", align_corners=False
        )

        return ref_feat_dense, match_conf_dense, ref_q_dense


class ConfidenceGatedFeatureTransfer(nn.Module):
    """
    Module D: Confidence-Controlled Residual Feature Transfer (Phase 3).
    Implements the principled residual feature-fusion equation:
        F_fused = F_target + alpha * confidence * (F_reference - F_target)

    Properties:
    - Conservative & Bounded: alpha is bounded in [0, max_transfer] (default max 0.35).
    - Zero Drift: When reference matches target content, (F_reference - F_target) -> 0.
    - Continuous Fallback: When confidence -> 0, F_fused = F_target strictly,
      completely preventing ghosting, blur, or dark blotchy artifacts.
    - Target need weighting: only low-SNR target pixels (1 - Q_target) receive assistance.
    """

    def __init__(
        self,
        feature_channels: int = 32,
        max_transfer: float = 0.35,
        init_alpha: float = 0.20,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.max_transfer = max_transfer

        # Learnable transfer strength parameter initialized conservatively
        self.alpha_param = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        # Spatial smoothing convolution without bias to guarantee zero-drift when feat_ref == feat_target
        self.diff_smooth = DepthwiseSeparableConv(
            feature_channels, feature_channels, kernel_size=3, padding=1, bias=False
        )


    def forward(
        self,
        feat_target: torch.Tensor,
        feat_ref: torch.Tensor,
        q_target: torch.Tensor,
        q_ref: torch.Tensor,
        match_conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_target: Target features (B, C, H, W)
            feat_ref: Transferred reference features (B, C, H, W)
            q_target: Target reliability map (B, 1, H, W)
            q_ref: Reference reliability map (B, 1, H, W)
            match_conf: Match confidence map (B, 1, H, W)
        Returns:
            feat_fused: Fused feature representation (B, C, H, W)
            effective_gate: Effective fusion gate map (B, 1, H, W)
        """
        # Bounded transfer coefficient in [0, max_transfer]
        alpha = torch.sigmoid(self.alpha_param) * self.max_transfer

        # Differential feature residual
        feat_diff = self.diff_smooth(feat_ref - feat_target)

        # Confidence weighting: structural confidence * reference quality * target unreliability
        need_factor = 1.0 - 0.75 * torch.clamp(q_target, 0.0, 1.0)
        # match_conf already includes reference reliability in InternalReferenceMatcher.
        # Multiplying q_ref again would square the reliability weighting.
        confidence = match_conf * need_factor
        effective_gate = alpha * confidence

        # Residual fusion equation: F_target + alpha * confidence * (F_reference - F_target)
        feat_fused = feat_target + effective_gate * feat_diff

        return feat_fused, effective_gate


class ConservativeEnhancementHead(nn.Module):
    """
    Module E: Conservative Chromaticity-Preserving Enhancement Head (Phase 5).
    1. Illumination Map L in [eps, 1.0].
    2. Base Retinex luminance correction: Y_base = Y_in / (L^gamma + eps).
    3. Proportional Chrominance Scaling:
       CbCr_base = CbCr_in * (Y_base / (Y_in + eps))^0.65
       Preserves vivid, authentic chromaticity without muted grey/beige washouts.
    4. Bounded residual refinement in YCbCr:
       Y_out = clamp(Y_base + alpha * (1 - 0.5 Q) * Delta_Y, 0.0, 1.0)
       CbCr_out = CbCr_base + beta * (1 - 0.5 Q) * Delta_C
       I_enh = ycbcr_to_rgb(Y_out, CbCr_out)
    """

    def __init__(
        self,
        feature_channels: int = 32,
        base_channels: int = 24,
        gamma: float = 0.80,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.eps = eps

        # Illumination estimator head from fused features
        self.head_illum = nn.Sequential(
            DepthwiseSeparableConv(feature_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Learnable residual scale parameters
        self.alpha = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.06, dtype=torch.float32))

        # Luminance residual head (predicts Delta_Y)
        self.head_lum = nn.Sequential(
            DepthwiseSeparableConv(feature_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Tanh(),
        )

        # Chrominance residual head (predicts Delta_CbCr)
        self.head_chrom = nn.Sequential(
            DepthwiseSeparableConv(feature_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(base_channels, 2, kernel_size=1),
            nn.Tanh(),
        )

    @staticmethod
    def adaptive_chrominance_gain(
        y_orig: torch.Tensor,
        y_base: torch.Tensor,
        q_map: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """Restore chroma in proportion to illumination recovery, gated by reliability.

        The gain is bounded by 6**0.70 (~3.51). A nonzero reliability floor
        allows some color recovery in dark regions, while higher-confidence
        pixels receive the full gain. q_map is detached so the model cannot
        reduce this chroma correction by gaming its reliability prediction.
        """
        illumination_ratio = ((y_base + eps) / (y_orig + eps)).clamp(1.0, 6.0)
        requested_gain = illumination_ratio.pow(0.70)
        reliability = q_map.detach().clamp(0.0, 1.0)
        chroma_confidence = 0.45 + 0.55 * reliability
        return 1.0 + chroma_confidence * (requested_gain - 1.0)

    def forward(
        self,
        feat_fused: torch.Tensor,
        original_rgb: torch.Tensor,
        q_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_fused: Fused feature map (B, C, H, W)
            original_rgb: Input RGB image (B, 3, H, W) in [0, 1]
            q_map: Reliability map (B, 1, H, W)
        Returns:
            enhanced_rgb: Enhanced RGB image (B, 3, H, W) in [0, 1]
            illumination: Illumination map L (B, 1, H, W) in [eps, 1.0]
            delta_y: Luminance residual (B, 1, H, W)
            delta_c: Chrominance residual (B, 2, H, W)
        """
        # 1. Illumination prediction
        illum = self.head_illum(feat_fused)
        illum = torch.clamp(illum, min=self.eps, max=1.0)

        # 2. Decompose input into YCbCr
        y_orig, cbcr_orig = ColorSpaceConverter.rgb_to_ycbcr(original_rgb)

        # 3. Retinex illumination correction on luminance
        illum_comp = torch.clamp(illum**self.gamma, min=self.eps, max=1.0)
        y_base = torch.clamp(y_orig / illum_comp, 0.0, 1.0)

        # 4. Predict residual corrections
        delta_y = self.head_lum(feat_fused)
        delta_c = self.head_chrom(feat_fused)

        # 5. Reliability-aware chrominance restoration.
        # Stronger than the previous sqrt(gain) rule, but bounded and attenuated
        # in unreliable pixels so lifting chroma does not blindly amplify noise.
        lum_gain = self.adaptive_chrominance_gain(
            y_orig=y_orig,
            y_base=y_base,
            q_map=q_map,
            eps=self.eps,
        )
        cbcr_scaled = cbcr_orig * lum_gain

        # 6. Reliability-modulated bounded updates
        mod_weight = 1.0 - 0.5 * torch.clamp(q_map, 0.0, 1.0)
        y_out = torch.clamp(y_base + torch.abs(self.alpha) * mod_weight * delta_y, 0.0, 1.0)
        cbcr_out = cbcr_scaled + self.beta * mod_weight * delta_c

        # 7. Gamut-aware inverse conversion. Uniformly reduce chroma only as
        # much as needed to keep RGB in range, avoiding per-channel clipping
        # that can shift hue and make restored colors look unnatural.
        enhanced_rgb = ColorSpaceConverter.ycbcr_to_rgb_gamut_safe(y_out, cbcr_out)

        return enhanced_rgb, illum, delta_y, delta_c


class SNRInternalReferenceNet(nn.Module):
    """
    SNR-Guided Internal Reference Transfer Network (v2).
    """

    def __init__(
        self,
        base_channels: int = 24,
        feature_channels: int = 32,
        num_blocks: int = 4,
        patch_size: int = 16,
        top_k: int = 3,
        min_ref_reliability: float = 0.40,
        similarity_threshold: float = 0.25,
        use_reference_transfer: bool = True,
        max_transfer: float = 0.35,
        act_type: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.use_reference_transfer = use_reference_transfer
        self.feature_channels = feature_channels

        # 1. Reliability & SNR-Proxy Estimator
        self.reliability_estimator = ReliabilityEstimator(
            base_channels=base_channels // 2,
        )

        # 2. Multi-Scale Feature Encoder
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(3, feature_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False) if act_type == "leaky_relu" else nn.GELU(),
        )
        dilations = [1, 2, 1, 2][:num_blocks]
        while len(dilations) < num_blocks:
            dilations.append(1)
        self.res_blocks = nn.ModuleList(
            [
                LightweightResBlock(
                    channels=feature_channels,
                    use_dw_conv=True,
                    dilation=dilations[i],
                    act_type=act_type,
                )
                for i in range(num_blocks)
            ]
        )

        # 3. Internal Reference Matcher
        self.reference_matcher = InternalReferenceMatcher(
            feature_channels=feature_channels,
            patch_size=patch_size,
            top_k=top_k,
            min_ref_reliability=min_ref_reliability,
            similarity_threshold=similarity_threshold,
        )

        # 4. Confidence-Gated Feature Transfer
        self.feature_transfer = ConfidenceGatedFeatureTransfer(
            feature_channels=feature_channels,
            max_transfer=max_transfer,
        )

        # 5. Enhancement Output Head
        self.enhancement_head = ConservativeEnhancementHead(
            feature_channels=feature_channels,
            base_channels=base_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard inference call returning enhanced RGB image."""
        diag = self.forward_diagnostic(x)
        return diag["enhanced"]

    def forward_diagnostic(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Full diagnostic forward pass providing all intermediate maps
        for loss computation and visualization.
        """
        lum = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

        # 1. Estimate Reliability / SNR-proxy map
        q_map, noise_proxy, sat_penalty = self.reliability_estimator(x, lum)

        # 2. Extract Multi-Scale Features
        feat = self.stem(x)
        for block in self.res_blocks:
            feat = block(feat)

        if self.use_reference_transfer:
            # 3. Extract high-reliability internal references & match
            ref_feat, match_conf, ref_q = self.reference_matcher(feat, q_map)

            # 4. Confidence-gated feature transfer
            feat_fused, transfer_gate = self.feature_transfer(
                feat, ref_feat, q_map, ref_q, match_conf
            )
        else:
            # Ablation path: reference transfer disabled
            b, _, h, w = feat.shape
            ref_feat = torch.zeros_like(feat)
            match_conf = torch.zeros((b, 1, h, w), device=x.device)
            ref_q = torch.zeros((b, 1, h, w), device=x.device)
            transfer_gate = torch.zeros((b, 1, h, w), device=x.device)
            feat_fused = feat

        # 5. Predict illumination and bounded enhancement
        enhanced, illumination, delta_y, delta_c = self.enhancement_head(
            feat_fused, x, q_map
        )

        return {
            "enhanced": enhanced,
            "illumination": illumination,
            "reliability_map": q_map,
            "noise_proxy": noise_proxy,
            "saturation_penalty": sat_penalty,
            "match_confidence": match_conf,
            "ref_reliability": ref_q,
            "transfer_gate": transfer_gate,
            "features": feat,
            "features_fused": feat_fused,
            "delta_y": delta_y,
            "delta_c": delta_c,
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_snr_internal_reference_net(
    config: Optional[Dict[str, Any]] = None,
) -> SNRInternalReferenceNet:
    """Factory helper to build SNRInternalReferenceNet from config."""
    if config is None:
        return SNRInternalReferenceNet()

    model_cfg = config.get("model", {})
    return SNRInternalReferenceNet(
        base_channels=model_cfg.get("base_channels", 24),
        feature_channels=model_cfg.get("feature_channels", 32),
        num_blocks=model_cfg.get("num_blocks", 4),
        patch_size=model_cfg.get("patch_size", 16),
        top_k=model_cfg.get("top_k", 3),
        min_ref_reliability=model_cfg.get("min_ref_reliability", 0.40),
        similarity_threshold=model_cfg.get("similarity_threshold", 0.25),
        use_reference_transfer=model_cfg.get("use_reference_transfer", True),
        max_transfer=model_cfg.get("max_transfer", 0.35),
        act_type=model_cfg.get("act_type", "leaky_relu"),
    )
