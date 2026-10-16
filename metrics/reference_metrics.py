"""
Reference-Based Evaluation Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- LPIPS (Learned Perceptual Image Patch Similarity - optional)
"""

from typing import Dict, Optional
import math
import torch
import torch.nn.functional as F
import numpy as np
#import lpips 


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Squared Error in [0, 1]."""
    return torch.mean((pred - target) ** 2).item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error in [0, 1]."""
    return torch.mean(torch.abs(pred - target)).item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = compute_mse(pred, target)
    if mse <= 1e-10:
        return 100.0
    return 10.0 * math.log10((max_val**2) / mse)


def compute_ssim_tensor(
    pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5
) -> float:
    """Computes mean SSIM across batches and channels."""
    channel = pred.size(1)
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.repeat(channel, 1, 1, 1).to(pred.device)

    c1 = 0.01**2
    c2 = 0.03**2

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


class ReferenceMetricsEvaluator:
    """Evaluates all reference metrics on batches of enhanced and ground-truth images."""

    def __init__(self, use_lpips: bool = False, device: str = "cpu") -> None:
        self.use_lpips = use_lpips
        self.lpips_fn = None

        if use_lpips:
            try:
                import lpips

                self.lpips_fn = lpips.LPIPS(net="alex").to(device)
                self.lpips_fn.eval()
            except ImportError:
                print("Notice: lpips package not installed. Skipping LPIPS calculation.")

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Args:
            pred: Enhanced image tensor (B, 3, H, W) in [0, 1]
            target: Ground truth image tensor (B, 3, H, W) in [0, 1]
        Returns:
            Dict containing PSNR, SSIM, MSE, MAE, and optional LPIPS.
        """
        metrics = {
            "psnr": compute_psnr(pred, target),
            "ssim": compute_ssim_tensor(pred, target),
            "mse": compute_mse(pred, target),
            "mae": compute_mae(pred, target),
        }

        if self.lpips_fn is not None:
            with torch.no_grad():
                # LPIPS expects input in [-1, 1]
                p_norm = pred * 2.0 - 1.0
                t_norm = target * 2.0 - 1.0
                lp = self.lpips_fn(p_norm, t_norm).mean().item()
                metrics["lpips"] = lp

        return metrics
