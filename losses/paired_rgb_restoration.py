"""Differentiable paired losses for low-light RGB restoration."""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def rgb_to_y_cbcr(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, torch.cat([cb, cr], dim=1)


def paired_rgb_restoration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """RGB fidelity plus restrained luminance, chroma, edge and local-contrast losses."""
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    pred_y, pred_cbcr = rgb_to_y_cbcr(prediction)
    target_y, target_cbcr = rgb_to_y_cbcr(target)

    loss_rgb = F.l1_loss(prediction, target)
    loss_luminance = F.l1_loss(pred_y, target_y)
    loss_chroma = F.l1_loss(pred_cbcr, target_cbcr)

    if prediction.shape[-1] > 1:
        pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        dx_loss = F.l1_loss(pred_dx, target_dx)
    else:
        dx_loss = prediction.new_zeros(())
    if prediction.shape[-2] > 1:
        pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        dy_loss = F.l1_loss(pred_dy, target_dy)
    else:
        dy_loss = prediction.new_zeros(())
    loss_gradient = 0.5 * (dx_loss + dy_loss)

    # Match patch-scale local luminance variability; handles full-size validation images.
    patch = min(16, prediction.shape[-2], prediction.shape[-1])
    def local_std(y: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(y, kernel_size=patch, stride=patch)
        mean_sq = F.avg_pool2d(y.square(), kernel_size=patch, stride=patch)
        return torch.sqrt((mean_sq - mean.square()).clamp_min(0.0) + 1e-6)

    loss_contrast = F.l1_loss(local_std(pred_y), local_std(target_y))

    total = (
        loss_rgb
        + 0.25 * loss_luminance
        + 0.25 * loss_chroma
        + 0.10 * loss_gradient
        + 0.10 * loss_contrast
    )
    return total, {
        "loss_total": total,
        "loss_rgb": loss_rgb,
        "loss_luminance": loss_luminance,
        "loss_chroma": loss_chroma,
        "loss_gradient": loss_gradient,
        "loss_contrast": loss_contrast,
    }
