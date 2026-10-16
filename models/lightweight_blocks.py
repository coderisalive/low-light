"""
Module B: Compact Multi-Scale Feature Extractor
Supports:
- Depthwise-separable convolutions
- Standard convolutions (ablation toggle)
- Configurable blocks (3-5), channels, and optional dilation
- Residual connections with LeakyReLU or GELU
"""

from typing import List, Optional
import torch
import torch.nn as nn
from models.illumination_noise_estimator import DepthwiseSeparableConv


class LightweightResBlock(nn.Module):
    """
    Lightweight residual block.
    Configurable between Depthwise-Separable convolution and standard convolution.
    """

    def __init__(
        self,
        channels: int,
        use_dw_conv: bool = True,
        dilation: int = 1,
        act_type: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.use_dw_conv = use_dw_conv

        if act_type == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.LeakyReLU(0.2, inplace=False)

        padding = dilation

        if use_dw_conv:
            self.conv1 = DepthwiseSeparableConv(
                channels, channels, kernel_size=3, padding=padding, dilation=dilation
            )
            self.conv2 = DepthwiseSeparableConv(
                channels, channels, kernel_size=3, padding=1, dilation=1
            )
        else:
            self.conv1 = nn.Conv2d(
                channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=True
            )
            self.conv2 = nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, dilation=1, bias=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + res


class CompactMultiScaleFeatureExtractor(nn.Module):
    """
    Compact multi-scale feature extractor.
    Extracts deep representation from the input image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_channels: int = 32,
        num_blocks: int = 4,
        use_dw_conv: bool = True,
        dilations: Optional[List[int]] = None,
        act_type: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.num_blocks = num_blocks

        # Input stem
        if use_dw_conv:
            self.stem = nn.Sequential(
                DepthwiseSeparableConv(in_channels, feature_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False) if act_type == "leaky_relu" else nn.GELU(),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=False) if act_type == "leaky_relu" else nn.GELU(),
            )

        if dilations is None:
            # Default multi-scale receptive field via cyclic dilations: [1, 2, 1, 2]
            dilations = [1, 2, 1, 2][:num_blocks]
            while len(dilations) < num_blocks:
                dilations.append(1)

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                LightweightResBlock(
                    channels=feature_channels,
                    use_dw_conv=use_dw_conv,
                    dilation=dilations[i],
                    act_type=act_type,
                )
            )
        self.res_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        for block in self.res_blocks:
            feat = block(feat)
        return feat
