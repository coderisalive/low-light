"""
Diagnostic Tool for SNR-Guided LumaGuard-Net.
Extracts and visualizes:
1. Input image statistics (mean, std, dynamic range, clipping)
2. Enhanced output statistics
3. Reliability map Q distribution and edge correlation
4. Reference matching confidence and patch locations
5. Illumination map smoothness and bounds
6. Feature norms before and after transfer
7. Paired metrics (PSNR, SSIM, MSE, MAE) against ground truth
"""

import argparse
from pathlib import Path
import os
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from models.snr_internal_reference_net import (
    SNRInternalReferenceNet,
    build_snr_internal_reference_net,
)
from models.lumaguard_net import LumaGuardNet, build_lumaguard_net
from metrics.reference_metrics import ReferenceMetricsEvaluator


def compute_image_stats(tensor: torch.Tensor, name: str) -> dict:
    """Compute numerical distribution metrics for an image tensor (1, 3, H, W) in [0, 1]."""
    t_np = tensor.squeeze(0).cpu().numpy()
    lum = 0.299 * t_np[0] + 0.587 * t_np[1] + 0.114 * t_np[2]

    stats = {
        f"{name}_min": float(np.min(t_np)),
        f"{name}_max": float(np.max(t_np)),
        f"{name}_mean": float(np.mean(lum)),
        f"{name}_std": float(np.std(lum)),
        f"{name}_clip_dark": float(np.mean(lum < 0.02) * 100),
        f"{name}_clip_bright": float(np.mean(lum > 0.98) * 100),
    }
    return stats


def diagnose_sample(
    model: torch.nn.Module,
    low_path: Path,
    high_path: Path,
    out_dir: Path,
    evaluator: ReferenceMetricsEvaluator,
    device: torch.device,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = low_path.stem

    with Image.open(low_path) as pil_low:
        pil_low = pil_low.convert("RGB")
        w, h = pil_low.size

    with Image.open(high_path) as pil_high:
        pil_high = pil_high.convert("RGB")
        pil_high = pil_high.resize((w, h), Image.Resampling.LANCZOS)

    # Pad to multiple of 16
    pad_w = (16 - (w % 16)) % 16
    pad_h = (16 - (h % 16)) % 16

    low_tensor = TF.to_tensor(pil_low).unsqueeze(0).to(device)
    high_tensor = TF.to_tensor(pil_high).unsqueeze(0).to(device)

    if pad_w > 0 or pad_h > 0:
        low_pad = torch.nn.functional.pad(low_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        low_pad = low_tensor

    model.eval()
    with torch.no_grad():
        diag = model.forward_diagnostic(low_pad)

        # Unpad
        for k in list(diag.keys()):
            if diag[k] is not None and isinstance(diag[k], torch.Tensor) and diag[k].dim() == 4:
                diag[k] = diag[k][:, :, :h, :w]

    enhanced = diag["enhanced"]

    # Compute metrics against GT
    metrics = evaluator.evaluate(enhanced, high_tensor)

    # Compute stats
    in_stats = compute_image_stats(low_tensor, "in")
    out_stats = compute_image_stats(enhanced, "out")
    gt_stats = compute_image_stats(high_tensor, "gt")

    # Feature norms before and after fusion
    feat_before = diag.get("features")
    feat_after = diag.get("features_fused")
    norm_delta = 0.0
    if feat_before is not None and feat_after is not None:
        norm_delta = float(torch.mean(torch.abs(feat_after - feat_before)).item())

    # Save diagnostic images
    # 1. Enhanced RGB
    enh_pil = TF.to_pil_image(enhanced.squeeze(0).cpu().clamp(0.0, 1.0))
    enh_pil.save(out_dir / f"{base_name}_enhanced.png")

    # 2. Reliability Map Q
    if diag.get("reliability_map") is not None:
        q_vis = diag["reliability_map"].squeeze(0).cpu().repeat(3, 1, 1).clamp(0.0, 1.0)
        TF.to_pil_image(q_vis).save(out_dir / f"{base_name}_reliability_map.png")

    # 3. Match Confidence Map
    if diag.get("match_confidence") is not None:
        conf_vis = diag["match_confidence"].squeeze(0).cpu().repeat(3, 1, 1).clamp(0.0, 1.0)
        TF.to_pil_image(conf_vis).save(out_dir / f"{base_name}_match_confidence.png")

    # 4. Illumination Map L
    if diag.get("illumination") is not None:
        l_vis = diag["illumination"].squeeze(0).cpu().repeat(3, 1, 1).clamp(0.0, 1.0)
        TF.to_pil_image(l_vis).save(out_dir / f"{base_name}_illumination_map.png")

    # 5. Difference Map
    diff = torch.abs(enhanced - low_tensor).squeeze(0).cpu().clamp(0.0, 1.0)
    TF.to_pil_image(diff).save(out_dir / f"{base_name}_diff_map.png")

    # 6. Triplet Comparison: [Input | Enhanced | Ground Truth]
    comp = torch.cat([low_tensor.squeeze(0).cpu(), enhanced.squeeze(0).cpu(), high_tensor.squeeze(0).cpu()], dim=2)
    TF.to_pil_image(comp.clamp(0.0, 1.0)).save(out_dir / f"{base_name}_triplet_comparison.png")

    summary = {
        "sample": base_name,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "mae": metrics["mae"],
        "feat_transfer_delta": norm_delta,
        **in_stats,
        **out_stats,
        **gt_stats,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Diagnose SNR-Guided LumaGuard-Net")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--low_dir", type=str, default="data/lol_dataset/eval15/low")
    parser.add_argument("--high_dir", type=str, default="data/lol_dataset/eval15/high")
    parser.add_argument("--output", type=str, default="outputs/diagnostics")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    config = ckpt.get("config", None)
    model_type = ckpt.get("model_type", None)

    is_snr = (model_type == "snr_internal_reference") or any(
        "reference_matcher" in k or "feature_transfer" in k for k in state_dict.keys()
    )

    if is_snr:
        model = build_snr_internal_reference_net(config).to(device)
    else:
        model = build_lumaguard_net(config).to(device)

    model.load_state_dict(state_dict)
    print(f"[Diagnostic] Loaded {type(model).__name__} from {args.checkpoint}")

    low_files = sorted(list(Path(args.low_dir).glob("*.png")))[:3]  # Evaluate first 3 representative images
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    for low_file in low_files:
        high_file = Path(args.high_dir) / low_file.name
        if not high_file.exists():
            continue
        stats = diagnose_sample(model, low_file, high_file, out_dir, evaluator, device)
        print(f"Sample: {stats['sample']}.png")
        print(f"  PSNR: {stats['psnr']:.2f} dB | SSIM: {stats['ssim']:.4f} | MAE: {stats['mae']:.4f}")
        print(f"  Input Mean Lum: {stats['in_mean']:.3f} | Enhanced Mean Lum: {stats['out_mean']:.3f} | GT Mean Lum: {stats['gt_mean']:.3f}")
        print(f"  Dark Clipped Pixels (<2%): {stats['out_clip_dark']:.1f}% | Bright Clipped (>98%): {stats['out_clip_bright']:.1f}%")
        print(f"  Feature Transfer L1 Delta: {stats['feat_transfer_delta']:.4f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
