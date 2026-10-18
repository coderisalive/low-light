"""Evaluate PairedRGBRestorationNet across multiple low-light benchmarks.

Supports:
- Paired Benchmarks (Full-Reference: PSNR, SSIM, MSE, MAE, Residual Visualizations):
  1. LOL-v1 eval15 (15 pairs)
  2. LOL-v2-Real Test (100 pairs)
  3. LOL-v2-Synthetic Test (100 pairs)
- Unpaired Real-World Benchmark (No-Reference: Entropy, RMS Contrast, Visual Inspection):
  4. ExDark Test (Unpaired low-light, NO fabricated ground truth)
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from datasets.multi_dataset import (
    SinglePairedDataset,
    build_official_test_datasets,
)
from datasets.exdark_dataset import ExDarkDataset
from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from metrics.reference_metrics import ReferenceMetricsEvaluator
from metrics.no_reference_metrics import evaluate_no_reference_metrics
from models.paired_rgb_restoration_net import PairedRGBRestorationNet


def to_rgb8(x: torch.Tensor) -> np.ndarray:
    return (x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def save_montage(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    images = [Image.fromarray(array.astype(np.uint8), mode="RGB") for _, array in panels]
    if not images or any(image.size != images[0].size for image in images):
        raise ValueError("Montage images must have identical dimensions.")
    w, h = images[0].size
    label_h = 34
    canvas = Image.new("RGB", (w * len(images), h + label_h), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, ((label, _), img) in enumerate(zip(panels, images)):
        x = i * w
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        draw.text((x + max(3, (w - tw) // 2), (label_h - th) // 2 - box[1]),
                  label, fill="white", font=font)
        canvas.paste(img, (x, label_h))
        if i:
            draw.line((x, 0, x, h + label_h), fill=(95, 95, 95), width=1)
    canvas.save(path)


def load_model(checkpoint_path: Path, device: torch.device):
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    model = PairedRGBRestorationNet(base_channels=int(cfg.get("base_channels", 24))).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def evaluate_paired_dataset(
    model: PairedRGBRestorationNet,
    loader: DataLoader,
    evaluator: ReferenceMetricsEvaluator,
    device: torch.device,
    ds_name: str,
    comparison_dir: Path,
    max_visuals: int = 8,
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, batch in enumerate(tqdm(loader, desc=f"Evaluating {ds_name}")):
        # SinglePairedDataset returns 4 items, PairedEvaluationDataset returns 3
        if len(batch) == 4:
            low, high, names, _ = batch
        else:
            low, high, names = batch

        low = low.to(device)
        high = high.to(device)
        output = model(low).clamp(0, 1)
        metrics = evaluator.evaluate(output, high.clamp(0, 1))
        name = names[0] if isinstance(names, (tuple, list)) else str(names)
        rows.append({"image": name, **metrics})

        if index < max_visuals:
            residual = ((output[0] - high[0]).abs() * 4.0).clamp(0, 1)
            save_montage(
                comparison_dir / f"{Path(name).stem}.png",
                [
                    (f"[{ds_name}] Low-light", to_rgb8(low[0])),
                    ("RGB U-Net Output", to_rgb8(output[0])),
                    ("Ground Truth", to_rgb8(high[0])),
                    ("Residual x4", to_rgb8(residual)),
                ],
            )

    summary = {key: float(np.mean([r[key] for r in rows])) for key in ("psnr", "ssim", "mse", "mae")}
    return summary, rows


@torch.inference_mode()
def evaluate_exdark_dataset(
    model: PairedRGBRestorationNet,
    loader: DataLoader,
    device: torch.device,
    comparison_dir: Path,
    max_visuals: int = 12,
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    entropies, contrasts = [], []

    for index, (low, cat, names, uids) in enumerate(tqdm(loader, desc="Evaluating ExDark (Unpaired)")):
        low = low.to(device)
        output = model(low).clamp(0, 1)
        m = evaluate_no_reference_metrics(output)
        name = names[0] if isinstance(names, (tuple, list)) else str(names)
        category = cat[0] if isinstance(cat, (tuple, list)) else str(cat)

        entropies.append(m["entropy"])
        contrasts.append(m["rms_contrast"])
        rows.append({
            "image": name,
            "category": category,
            "entropy": f"{m['entropy']:.4f}",
            "rms_contrast": f"{m['rms_contrast']:.4f}",
        })

        if index < max_visuals:
            save_montage(
                comparison_dir / f"{category}_{Path(name).stem}.png",
                [
                    (f"ExDark [{category}] Input", to_rgb8(low[0])),
                    ("RGB U-Net Output", to_rgb8(output[0])),
                ],
            )

    summary = {
        "entropy": float(np.mean(entropies)),
        "rms_contrast": float(np.mean(contrasts)),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PairedRGBRestorationNet across multiple benchmarks.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pth checkpoint.")
    # Single dataset flags (backwards compatible)
    parser.add_argument("--low_dir", default=None, help="Low-light directory for single-dataset evaluation.")
    parser.add_argument("--high_dir", default=None, help="Reference directory for single-dataset evaluation.")

    # Multi-dataset roots
    parser.add_argument("--lolv1_root", default=None, help="Root folder for LOL-v1 (eval15 test used).")
    parser.add_argument("--lolv2_real_root", default=None, help="Root folder for LOL-v2-Real (Real_captured Test used).")
    parser.add_argument("--lolv2_synthetic_root", default=None, help="Root folder for LOL-v2-Synthetic (Synthetic Test used).")
    parser.add_argument("--exdark_root", default=None, help="Root folder for ExDark (unpaired test split used).")

    parser.add_argument("--output_dir", default="experiments/multi_benchmark_eval", help="Destination folder for metrics and images.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max_visuals", type=int, default=8, help="Number of comparison figures to save per benchmark.")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    print(f"[Device] {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = load_model(checkpoint, device)
    print(f"[Model] PairedRGBRestorationNet | {model.count_parameters():,} parameters loaded from {checkpoint}")
    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_report_path = out_dir / "benchmark_summary.csv"

    # Case A: Legacy single dataset evaluation
    if args.low_dir and args.high_dir:
        low_dir = Path(args.low_dir)
        high_dir = Path(args.high_dir)
        ds = PairedEvaluationDataset(str(low_dir), str(high_dir))
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        comp_dir = out_dir / "comparisons"
        summary, rows = evaluate_paired_dataset(model, loader, evaluator, device, "Custom", comp_dir, args.max_visuals)

        csv_path = out_dir / "evaluation_per_image.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "psnr", "ssim", "mse", "mae"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n[Single Evaluation] {len(rows)} images | PSNR={summary['psnr']:.3f} dB | SSIM={summary['ssim']:.4f} | MAE={summary['mae']:.5f}")
        print(f"Results saved to: {csv_path}")
        return

    # Case B: Multi-Benchmark Evaluation
    roots_to_check = {}
    if args.lolv1_root:
        roots_to_check["lol_v1"] = Path(args.lolv1_root).resolve()
    if args.lolv2_real_root:
        roots_to_check["lol_v2_real"] = Path(args.lolv2_real_root).resolve()
    if args.lolv2_synthetic_root:
        roots_to_check["lol_v2_synthetic"] = Path(args.lolv2_synthetic_root).resolve()

    # Fallback auto-discovery if no multi-roots given
    if not roots_to_check and not args.exdark_root:
        candidates = [
            ("lol_v1", Path("data/lol_dataset")),
            ("lol_v1", Path("/kaggle/working/lol_compat")),
            ("lol_v2_real", Path("data/LOL-v2-Real")),
            ("lol_v2_synthetic", Path("data/LOL-v2-Synthetic")),
        ]
        for name, p in candidates:
            if p.is_dir() and name not in roots_to_check:
                roots_to_check[name] = p.resolve()

    test_datasets = build_official_test_datasets(roots_to_check)
    summary_rows = []

    print("\n" + "=" * 80)
    print("                MULTI-BENCHMARK EVALUATION LEADERBOARD")
    print("=" * 80)

    # 1. Evaluate Paired Benchmarks
    for ds_name, ds in test_datasets.items():
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        comp_dir = out_dir / "comparisons" / ds_name
        summary, rows = evaluate_paired_dataset(model, loader, evaluator, device, ds_name, comp_dir, args.max_visuals)

        # Save per-image CSV
        csv_path = out_dir / f"{ds_name}_test_per_image.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "psnr", "ssim", "mse", "mae"])
            writer.writeheader()
            writer.writerows(rows)

        summary_rows.append({
            "benchmark": ds_name.upper(),
            "type": "Paired (Full-Reference)",
            "num_images": len(rows),
            "psnr_or_entropy": f"{summary['psnr']:.2f} dB",
            "ssim_or_contrast": f"{summary['ssim']:.4f}",
            "mae": f"{summary['mae']:.5f}",
            "details": f"Saved to {csv_path.name}",
        })
        print(f"[{ds_name.upper():<18}] {len(rows):>3} images | PSNR: {summary['psnr']:>6.2f} dB | SSIM: {summary['ssim']:>6.4f} | MAE: {summary['mae']:>7.5f}")

    # 2. Evaluate ExDark (Unpaired)
    exdark_path = args.exdark_root
    if not exdark_path:
        for cand in [Path("data/exdark"), Path("data/ExDark"), Path("/kaggle/input/exdark")]:
            if cand.is_dir():
                exdark_path = str(cand)
                break

    if exdark_path and Path(exdark_path).is_dir():
        exdark_ds = ExDarkDataset(exdark_path, split="test", crop_size=None, augment=False)
        if len(exdark_ds) > 0:
            loader = DataLoader(exdark_ds, batch_size=1, shuffle=False)
            comp_dir = out_dir / "comparisons" / "exdark"
            exdark_summary, exdark_rows = evaluate_exdark_dataset(model, loader, device, comp_dir, args.max_visuals)

            csv_path = out_dir / "exdark_unpaired_metrics.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "category", "entropy", "rms_contrast"])
                writer.writeheader()
                writer.writerows(exdark_rows)

            summary_rows.append({
                "benchmark": "EXDARK",
                "type": "Unpaired (No-Reference)",
                "num_images": len(exdark_rows),
                "psnr_or_entropy": f"Entropy: {exdark_summary['entropy']:.3f}",
                "ssim_or_contrast": f"RMS: {exdark_summary['rms_contrast']:.4f}",
                "mae": "N/A (No GT)",
                "details": f"Saved to {csv_path.name}",
            })
            print(f"[{'EXDARK (UNPAIRED)':<18}] {len(exdark_rows):>3} images | Entropy: {exdark_summary['entropy']:>6.3f} | RMS Contrast: {exdark_summary['rms_contrast']:>6.4f} | (No GT fabricated)")

    print("=" * 80)

    # Save summary table
    if summary_rows:
        with open(summary_report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["benchmark", "type", "num_images", "psnr_or_entropy", "ssim_or_contrast", "mae", "details"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[Summary Report Saved] {summary_report_path.resolve()}\n")
    else:
        print("[Warning] No test datasets were found to evaluate. Provide --lolv1_root, --lolv2_real_root, --lolv2_synthetic_root, or --exdark_root.")


if __name__ == "__main__":
    main()
