"""
Module C: Reliability-Guided Feature Modulation
Estimates illumination-noise reliability map:
    Q = sigmoid(a * log(L + eps) - b * N + c)
Learnable feature gate:
    G = sigmoid(Conv1x1(concat(F, L, N, Q)))
    F_guided = F + F * G

Includes:
- Stability clamp & gradient safeguards
- Ablation switches (disable reliability map, disable gate entirely)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class ReliabilityGate(nn.Module):
    """
    Reliability-Guided Feature Modulation module.
    Modulates intermediate features F using spatial illumination L and noise proxy N.
    """

    def __init__(
        self,
        feature_channels: int = 32,
        enabled: bool = True,
        use_reliability_map: bool = True,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.use_reliability_map = use_reliability_map
        self.eps = eps

        # Learnable parameters for reliability proxy Q:
        # Initial values: a=1.0, b=1.0, c=0.0
        self.param_a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.param_b = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.param_c = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Gate input channels:
        # F: feature_channels
        # L: 1
        # N: 1
        # Q: 1 (if used)
        extra_channels = 3 if use_reliability_map else 2
        in_channels = feature_channels + extra_channels

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels // 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(feature_channels // 2, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def compute_reliability_map(self, l_map: torch.Tensor, n_map: torch.Tensor) -> torch.Tensor:
        """
        Q = sigmoid(a * log(L + eps) - b * N + c)
        High reliability when L is high and N is low.
        """
        log_l = torch.log(torch.clamp(l_map, min=self.eps, max=1.0))
        # Clamp log_l to prevent negative overflow
        log_l = torch.clamp(log_l, min=-10.0, max=0.0)
        logit = self.param_a * log_l - self.param_b * n_map + self.param_c
        # Clamp logit before sigmoid for numerical stability
        logit = torch.clamp(logit, min=-15.0, max=15.0)
        return torch.sigmoid(logit)

    def forward(
        self,
        feat: torch.Tensor,
        l_map: torch.Tensor,
        n_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            feat: Feature map F of shape (B, C, H, W)
            l_map: Illumination map L of shape (B, 1, H, W)
            n_map: Noise proxy map N of shape (B, 1, H, W)
        Returns:
            feat_guided: Modulated feature map (B, C, H, W)
            q_map: Reliability map (B, 1, H, W) or None if disabled
            gate: Gate map (B, C, H, W) or None if disabled
        """
        if not self.enabled:
            return feat, None, None

        if self.use_reliability_map:
            q_map = self.compute_reliability_map(l_map, n_map)
            gate_in = torch.cat([feat, l_map, n_map, q_map], dim=1)
        else:
            q_map = None
            gate_in = torch.cat([feat, l_map, n_map], dim=1)

        gate = self.gate_conv(gate_in)
        # Guided modulation: residual amplification bounded by gate
        feat_guided = feat + feat * gate
        return feat_guided, q_map, gate
