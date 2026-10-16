"""
LumaGuard-Net: Lightweight Noise-Guided Self-Supervised Network for Low-Light Image Enhancement
Main Architecture assembling Modules A, B, C, and D.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from models.illumination_noise_estimator import IlluminationNoiseEstimator
from models.lightweight_blocks import CompactMultiScaleFeatureExtractor
from models.reliability_gate import ReliabilityGate
from models.enhancement_head import BoundedResidualEnhancementHead


class LumaGuardNet(nn.Module):
    """
    LumaGuard-Net Architecture.
    Four primary components:
    1. Illumination-Noise Estimator (INE)
    2. Compact Multi-Scale Feature Extractor
    3. Reliability-Guided Feature Modulation (Gate)
    4. Bounded Residual Enhancement Head
    """

    def __init__(
        self,
        base_channels: int = 24,
        feature_channels: int = 32,
        num_blocks: int = 4,
        use_dw_conv: bool = True,
        use_reliability_map: bool = True,
        use_reliability_gate: bool = True,
        use_noise_estimation: bool = True,
        use_separate_color: bool = True,
        act_type: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.use_noise_estimation = use_noise_estimation

        # Module A: Illumination-Noise Estimator
        self.estimator = IlluminationNoiseEstimator(
            in_channels=4,  # RGB (3) + Luminance (1)
            base_channels=base_channels,
        )

        # Module B: Multi-Scale Feature Extractor
        self.feature_extractor = CompactMultiScaleFeatureExtractor(
            in_channels=3,
            feature_channels=feature_channels,
            num_blocks=num_blocks,
            use_dw_conv=use_dw_conv,
            act_type=act_type,
        )

        # Module C: Reliability-Guided Feature Modulation
        self.reliability_gate = ReliabilityGate(
            feature_channels=feature_channels,
            enabled=use_reliability_gate,
            use_reliability_map=use_reliability_map,
        )

        # Module D: Bounded Residual Enhancement Head
        self.enhancement_head = BoundedResidualEnhancementHead(
            feature_channels=feature_channels,
            use_separate_color=use_separate_color,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple inference interface: returns enhanced RGB image.
        Args:
            x: Low-light image tensor (B, 3, H, W) normalized to [0, 1]
        Returns:
            enhanced: Enhanced RGB tensor (B, 3, H, W) in [0, 1]
        """
        diag = self.forward_diagnostic(x)
        return diag["enhanced"]

    def forward_diagnostic(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Structured forward pass returning diagnostics for research analysis and loss calculation.
        Args:
            x: Low-light input (B, 3, H, W)
        Returns:
            Dictionary with:
            - 'enhanced': Enhanced RGB image (B, 3, H, W)
            - 'illumination': Estimated illumination map L (B, 1, H, W)
            - 'noise_uncertainty': Estimated noise proxy N (B, 1, H, W)
            - 'reliability_map': Reliability map Q (B, 1, H, W) or None
            - 'gate': Gate tensor G (B, C, H, W) or None
            - 'features': Extracted features F before gating (B, C, H, W)
            - 'features_guided': Features F_guided after gating (B, C, H, W)
            - 'delta_y': Luminance correction (B, 1, H, W)
            - 'delta_c': Chrominance correction (B, 2, H, W)
        """
        # Module A: Illumination and Noise Estimation
        l_map, n_map = self.estimator(x)

        if not self.use_noise_estimation:
            # Ablation: zero out noise uncertainty map
            n_map = torch.zeros_like(n_map)

        # Module B: Multi-Scale Feature Extraction
        features = self.feature_extractor(x)

        # Module C: Reliability-Guided Modulation
        features_guided, q_map, gate = self.reliability_gate(features, l_map, n_map)

        # Module D: Enhancement Head
        enhanced, delta_y, delta_c = self.enhancement_head(features_guided, x)

        return {
            "enhanced": enhanced,
            "illumination": l_map,
            "noise_uncertainty": n_map,
            "reliability_map": q_map,
            "gate": gate,
            "features": features,
            "features_guided": features_guided,
            "delta_y": delta_y,
            "delta_c": delta_c,
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_lumaguard_net(config: Optional[Dict[str, Any]] = None) -> LumaGuardNet:
    """Factory helper to build LumaGuardNet from config dict."""
    if config is None:
        return LumaGuardNet()

    model_cfg = config.get("model", {})
    return LumaGuardNet(
        base_channels=model_cfg.get("base_channels", 24),
        feature_channels=model_cfg.get("feature_channels", 32),
        num_blocks=model_cfg.get("num_blocks", 4),
        use_dw_conv=model_cfg.get("use_dw_conv", True),
        use_reliability_map=model_cfg.get("use_reliability_map", True),
        use_reliability_gate=model_cfg.get("use_reliability_gate", True),
        use_noise_estimation=model_cfg.get("use_noise_estimation", True),
        use_separate_color=model_cfg.get("use_separate_color", True),
        act_type=model_cfg.get("act_type", "leaky_relu"),
    )
