"""
No-Reference Image Quality Metrics:
- NIQE (Naturalness Image Quality Evaluator) proxy / statistical naturalness
- Image Entropy & Contrast metrics

Scientific Note:
No-reference metrics do not perfectly measure perceptual quality and can be biased
towards high contrast, over-sharpening, or artificial noise patterns.
"""

from typing import Dict
import math
import numpy as np
import torch


def compute_image_entropy(img: torch.Tensor) -> float:
    """Computes Shannon entropy across the image luminance."""
    # img: (B, 3, H, W) in [0, 1]
    lum = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
    lum_np = (lum.detach().cpu().numpy() * 255.0).astype(np.uint8)

    entropies = []
    for b in range(lum_np.shape[0]):
        hist, _ = np.histogram(lum_np[b], bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        ent = -np.sum(hist * np.log2(hist))
        entropies.append(ent)
    return float(np.mean(entropies))


def compute_rms_contrast(img: torch.Tensor) -> float:
    """Computes Root Mean Square (RMS) luminance contrast."""
    lum = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
    std = torch.std(lum, dim=(-2, -1))
    return float(torch.mean(std).item())


def evaluate_no_reference_metrics(img: torch.Tensor) -> Dict[str, float]:
    """
    Evaluates statistical no-reference quality metrics.
    """
    return {
        "entropy": compute_image_entropy(img),
        "rms_contrast": compute_rms_contrast(img),
    }
