"""
Ablation and Diagnostic Script: Resolving the PSNR vs. SSIM Trade-off in SNR-Guided LumaGuard-Net.

Evaluates on identical eval15 benchmark images:
1. Baseline (LumaGuard-Net)
2. New Model w/o Internal Reference Transfer (use_reference_transfer=False)
3. Full New Model with Internal Reference Transfer (use_reference_transfer=True)

Computes per-image and aggregate metrics:
- PSNR, SSIM, MSE, MAE
- Edge Gradient Energy Ratio (High-Frequency Fidelity): ||grad(Pred)||_1 / ||grad(GT)||_1
- Chromatic Cosine Similarity (Color Angle Alignment)
- Mean Luminance Difference (Exposure Bias): Y_pred - Y_gt

Generates 5-panel side-by-side visual comparisons:
[Input Low | Ground Truth | Baseline | New Model (No Transfer) | New Model (Full)]
"""

import argparse
import os
import sys
from pathlib import Path
import glob
import csv
import math

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from tqdm import tqdm

from models.lumaguard_net import LumaGuardNet, build_lumaguard_net
from models.snr_internal_reference_net import (
    SNRInternalReferenceNet,
    build_snr_internal_reference_net,
)
from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from metrics.reference_metrics import compute_psnr, compute_ssim_tensor, compute_mse, compute_mae


def compute_edge_energy_ratio(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Computes ratio of high-frequency Sobel gradient magnitude:
    Ratio < 1.0 indicates oversmoothing/blurring;
    Ratio > 1.0 indicates noise amplification or haloing.
    """
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=pred.device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=pred.device).view(1, 1, 3, 3) / 8.0
    
    y_pred = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
    y_gt = 0.299 * gt[:, 0:1] + 0.587 * gt[:, 1:2] + 0.114 * gt[:, 2:3]
    
    gx_pred = F.conv2d(y_pred, sobel_x, padding=1)
    gy_pred = F.conv2d(y_pred, sobel_y, padding=1)
    mag_pred = torch.sqrt(gx_pred**2 + gy_pred**2 + 1e-8).mean().item()
    
    gx_gt = F.conv2d(y_gt, sobel_x, padding=1)
    gy_gt = F.conv2d(y_gt, sobel_y, padding=1)
    mag_gt = torch.sqrt(gx_gt**2 + gy_gt**2 + 1e-8).mean().item()
    
    return mag_pred / (mag_gt + 1e-8)


def compute_chromatic_cosine(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Measures cosine similarity of RGB vectors per pixel, averaged across image."""
    dot = torch.sum(pred * gt, dim=1, keepdim=True)
    norm_p = torch.sqrt(torch.sum(pred * pred, dim=1, keepdim=True) + 1e-8)
    norm_g = torch.sqrt(torch.sum(gt * gt, dim=1, keepdim=True) + 1e-8)
    cos = dot / (norm_p * norm_g)
    return cos.mean().item()


def compute_luminance_bias(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean luminance difference (pred - gt). Positive = brighter than GT, Negative = darker."""
    y_pred = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
    y_gt = 0.299 * gt[:, 0:1] + 0.587 * gt[:, 1:2] + 0.114 * gt[:, 2:3]
    return (y_pred.mean() - y_gt.mean()).item()


def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple_Pad:
    _, _, h, w = x.shape
    pad_w = (multiple - (w % multiple)) % multiple
    pad_h = (multiple - (h % multiple)) % multiple
    if pad_w > 0 or pad_h > 0:
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x_pad, h, w
    return x, h, w


Tuple_Pad = tuple


def run_inference(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x_pad, h, w = pad_to_multiple(x, 16)
    with torch.no_grad():
        out = model(x_pad)
    return out[:, :, :h, :w]


def find_checkpoint(explicit_path: Optional[str] = None, pattern: str = "best_checkpoint.pth") -> Optional[str]:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    
    candidates = [
        "experiments/snr_reference_kaggle_gpu/best_checkpoint.pth",
        "experiments/snr_reference_lol_v1/best_checkpoint.pth",
        "/kaggle/working/best_checkpoint.pth",
        "/kaggle/working/experiments/best_checkpoint.pth",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    
    # Recursive search in experiments and kaggle working
    found = glob.glob(f"experiments/**/{pattern}", recursive=True) + glob.glob(f"/kaggle/working/**/{pattern}", recursive=True)
    for f in found:
        if "lol_v1_10epochs" not in f:
            return f
    return None


def main():
    parser = argparse.ArgumentParser(description="Ablate SNR-Guided LumaGuard-Net Trade-off")
    parser.add_argument("--baseline_checkpoint", type=str, default="experiments/lol_v1_10epochs/best_checkpoint.pth")
    parser.add_argument("--new_model_checkpoint", type=str, default=None)
    parser.add_argument("--low_dir", type=str, default="data/lol_dataset/eval15/low")
    parser.add_argument("--high_dir", type=str, default="data/lol_dataset/eval15/high")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation_tradeoff")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"[Ablation Protocol] Running on device: {dev}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = out_dir / "comparisons"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Baseline Model
    baseline_model = None
    if os.path.exists(args.baseline_checkpoint):
        print(f"[Loading] Baseline checkpoint: {args.baseline_checkpoint}")
        ckpt_base = torch.load(args.baseline_checkpoint, map_location=dev)
        state_dict_b = ckpt_base["model"] if isinstance(ckpt_base, dict) and "model" in ckpt_base else ckpt_base
        config_b = ckpt_base.get("config", None) if isinstance(ckpt_base, dict) else None
        baseline_model = build_lumaguard_net(config_b).to(dev)
        baseline_model.load_state_dict(state_dict_b)
        baseline_model.eval()
        print(f"[Baseline] LumaGuardNet loaded successfully ({baseline_model.count_parameters():,} params)")
    else:
        print(f"[Warning] Baseline checkpoint not found at {args.baseline_checkpoint}")

    # 2. Locate and Load New Model Checkpoint
    new_ckpt_path = find_checkpoint(args.new_model_checkpoint)
    new_model_full = None
    new_model_notransfer = None

    if new_ckpt_path and os.path.exists(new_ckpt_path):
        print(f"[Loading] New model checkpoint: {new_ckpt_path}")
        ckpt_new = torch.load(new_ckpt_path, map_location=dev)
        state_dict_n = ckpt_new["model"] if isinstance(ckpt_new, dict) and "model" in ckpt_new else ckpt_new
        config_n = ckpt_new.get("config", None) if isinstance(ckpt_new, dict) else None

        # Build Full Model (transfer enabled)
        new_model_full = build_snr_internal_reference_net(config_n).to(dev)
        new_model_full.load_state_dict(state_dict_n)
        new_model_full.use_reference_transfer = True
        new_model_full.eval()

        # Build No-Transfer Model (transfer disabled, using exact same weights)
        new_model_notransfer = build_snr_internal_reference_net(config_n).to(dev)
        new_model_notransfer.load_state_dict(state_dict_n)
        new_model_notransfer.use_reference_transfer = False
        new_model_notransfer.eval()

        print(f"[New Model] SNRInternalReferenceNet loaded successfully ({new_model_full.count_parameters():,} params)")
    else:
        print(f"[Notice] New model checkpoint not found locally. Searching Kaggle working directories...")

    # 3. Load Dataset
    if not (os.path.exists(args.low_dir) and os.path.exists(args.high_dir)):
        print(f"[Error] Dataset directories do not exist: {args.low_dir}, {args.high_dir}")
        return

    dataset = PairedEvaluationDataset(args.low_dir, args.high_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"[Dataset] Evaluating on {len(dataset)} paired test images from {args.low_dir}")

    # Results dictionaries
    records = []
    summary_metrics = {
        "Baseline": {"psnr": [], "ssim": [], "mse": [], "edge_ratio": [], "chrom_cos": [], "lum_bias": []},
        "No-Transfer": {"psnr": [], "ssim": [], "mse": [], "edge_ratio": [], "chrom_cos": [], "lum_bias": []},
        "Full-Model": {"psnr": [], "ssim": [], "mse": [], "edge_ratio": [], "chrom_cos": [], "lum_bias": []},
    }

    for low, high, img_name in tqdm(loader, desc="Evaluating Ablations"):
        low = low.to(dev)
        high = high.to(dev)
        name_str = img_name[0] if isinstance(img_name, (list, tuple)) else str(img_name)

        row = {"image": name_str}
        enh_base = None
        enh_notransfer = None
        enh_full = None

        # Condition 1: Baseline
        if baseline_model is not None:
            enh_base = run_inference(baseline_model, low)
            p_b = compute_psnr(enh_base, high)
            s_b = compute_ssim_tensor(enh_base, high)
            m_b = compute_mse(enh_base, high)
            e_b = compute_edge_energy_ratio(enh_base, high)
            c_b = compute_chromatic_cosine(enh_base, high)
            l_b = compute_luminance_bias(enh_base, high)

            summary_metrics["Baseline"]["psnr"].append(p_b)
            summary_metrics["Baseline"]["ssim"].append(s_b)
            summary_metrics["Baseline"]["mse"].append(m_b)
            summary_metrics["Baseline"]["edge_ratio"].append(e_b)
            summary_metrics["Baseline"]["chrom_cos"].append(c_b)
            summary_metrics["Baseline"]["lum_bias"].append(l_b)

            row["baseline_psnr"] = f"{p_b:.2f}"
            row["baseline_ssim"] = f"{s_b:.4f}"
            row["baseline_edge_ratio"] = f"{e_b:.3f}"

        # Condition 2: New Model (No Reference Transfer)
        if new_model_notransfer is not None:
            enh_notransfer = run_inference(new_model_notransfer, low)
            p_nt = compute_psnr(enh_notransfer, high)
            s_nt = compute_ssim_tensor(enh_notransfer, high)
            m_nt = compute_mse(enh_notransfer, high)
            e_nt = compute_edge_energy_ratio(enh_notransfer, high)
            c_nt = compute_chromatic_cosine(enh_notransfer, high)
            l_nt = compute_luminance_bias(enh_notransfer, high)

            summary_metrics["No-Transfer"]["psnr"].append(p_nt)
            summary_metrics["No-Transfer"]["ssim"].append(s_nt)
            summary_metrics["No-Transfer"]["mse"].append(m_nt)
            summary_metrics["No-Transfer"]["edge_ratio"].append(e_nt)
            summary_metrics["No-Transfer"]["chrom_cos"].append(c_nt)
            summary_metrics["No-Transfer"]["lum_bias"].append(l_nt)

            row["notransfer_psnr"] = f"{p_nt:.2f}"
            row["notransfer_ssim"] = f"{s_nt:.4f}"
            row["notransfer_edge_ratio"] = f"{e_nt:.3f}"

        # Condition 3: Full New Model (With Confidence-Controlled Transfer)
        if new_model_full is not None:
            enh_full = run_inference(new_model_full, low)
            p_f = compute_psnr(enh_full, high)
            s_f = compute_ssim_tensor(enh_full, high)
            m_f = compute_mse(enh_full, high)
            e_f = compute_edge_energy_ratio(enh_full, high)
            c_f = compute_chromatic_cosine(enh_full, high)
            l_f = compute_luminance_bias(enh_full, high)

            summary_metrics["Full-Model"]["psnr"].append(p_f)
            summary_metrics["Full-Model"]["ssim"].append(s_f)
            summary_metrics["Full-Model"]["mse"].append(m_f)
            summary_metrics["Full-Model"]["edge_ratio"].append(e_f)
            summary_metrics["Full-Model"]["chrom_cos"].append(c_f)
            summary_metrics["Full-Model"]["lum_bias"].append(l_f)

            row["full_psnr"] = f"{p_f:.2f}"
            row["full_ssim"] = f"{s_f:.4f}"
            row["full_edge_ratio"] = f"{e_f:.3f}"

        records.append(row)

        # Generate 5-way side-by-side comparison image
        panels = [low.squeeze(0).cpu(), high.squeeze(0).cpu()]
        if enh_base is not None:
            panels.append(enh_base.squeeze(0).cpu())
        if enh_notransfer is not None:
            panels.append(enh_notransfer.squeeze(0).cpu())
        if enh_full is not None:
            panels.append(enh_full.squeeze(0).cpu())

        grid = vutils.make_grid(panels, nrow=len(panels), padding=4, normalize=False)
        vutils.save_image(grid, visuals_dir / f"{Path(name_str).stem}_ablation.png")

    # Export Per-Image CSV
    csv_path = out_dir / "ablation_per_image_results.csv"
    if records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"[Saved] Per-image ablation table saved to: {csv_path}")

    # Print Summary Tables
    print("\n" + "=" * 90)
    print(f"{'Condition':<25} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'MSE':<10} | {'Edge Ratio':<12} | {'Lum Bias'}")
    print("=" * 90)

    for cond, m in summary_metrics.items():
        if len(m["psnr"]) > 0:
            mean_p = np.mean(m["psnr"])
            mean_s = np.mean(m["ssim"])
            mean_m = np.mean(m["mse"])
            mean_e = np.mean(m["edge_ratio"])
            mean_l = np.mean(m["lum_bias"])
            print(f"{cond:<25} | {mean_p:>10.2f} dB | {mean_s:>8.4f}   | {mean_m:>8.4f}   | {mean_e:>10.3f}   | {mean_l:>+8.4f}")

    print("=" * 90)

    # Automated Diagnosis & Recommendation
    if len(summary_metrics["Full-Model"]["psnr"]) > 0 and len(summary_metrics["No-Transfer"]["psnr"]) > 0:
        p_diff = np.mean(summary_metrics["Full-Model"]["psnr"]) - np.mean(summary_metrics["No-Transfer"]["psnr"])
        s_diff = np.mean(summary_metrics["Full-Model"]["ssim"]) - np.mean(summary_metrics["No-Transfer"]["ssim"])
        e_diff = np.mean(summary_metrics["Full-Model"]["edge_ratio"]) - np.mean(summary_metrics["No-Transfer"]["edge_ratio"])

        print("\n[Ablation Diagnostic Analysis]")
        print(f"- Effect of Reference Transfer on PSNR: {p_diff:+.3f} dB")
        print(f"- Effect of Reference Transfer on SSIM: {s_diff:+.4f}")
        print(f"- Effect on Edge High-Frequency Energy: {e_diff:+.4f}")

        if s_diff < -0.005 and p_diff > 0.5:
            print("[Diagnosis]: Reference transfer is improving global exposure alignment (higher PSNR / lower MSE),")
            print("but introducing micro-structural smoothing or spatial mismatch at patch boundaries (lower SSIM).")
        elif s_diff > 0.005:
            print("[Diagnosis]: Reference transfer successfully enhances structural similarity and fine details.")


if __name__ == "__main__":
    main()
