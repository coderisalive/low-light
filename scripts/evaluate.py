"""
Evaluation script for paired and unpaired low-light datasets.
Computes:
- PSNR, SSIM, MSE, MAE, LPIPS (on paired sets)
- Entropy and RMS Contrast (on unpaired sets)
- Exports results to CSV
"""

import argparse
import os
import sys
from pathlib import Path
import csv

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.lumaguard_net import LumaGuardNet, build_lumaguard_net
from models.snr_internal_reference_net import (
    SNRInternalReferenceNet,
    build_snr_internal_reference_net,
)
from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from datasets.exdark_dataset import ExDarkDataset
from metrics.reference_metrics import ReferenceMetricsEvaluator
from metrics.no_reference_metrics import evaluate_no_reference_metrics


def evaluate_paired(model, val_loader, evaluator, device, comparison_dir=None):
    model.eval()
    results = {"psnr": [], "ssim": [], "mse": [], "mae": []}
    per_image = []
    if comparison_dir is not None:
        comparison_dir = Path(comparison_dir)
        comparison_dir.mkdir(parents=True, exist_ok=True)

    for low, high, name in tqdm(val_loader, desc="Evaluating Paired"):
        low = low.to(device)
        high = high.to(device)

        with torch.no_grad():
            _, _, h, w = low.shape
            pad_w = (16 - (w % 16)) % 16
            pad_h = (16 - (h % 16)) % 16
            if pad_w > 0 or pad_h > 0:
                low_pad = torch.nn.functional.pad(low, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                low_pad = low
            enhanced = model(low_pad)[:, :, :h, :w].clamp(0.0, 1.0)
            metrics = evaluator.evaluate(enhanced, high.clamp(0.0, 1.0))
            image_name = name[0] if isinstance(name, (tuple, list)) else str(name)
            per_image.append({"image": image_name, **metrics})
            for key in results:
                results[key].append(metrics[key])

            if comparison_dir is not None:
                residual = ((enhanced - high).abs() * 4.0).clamp(0.0, 1.0)
                tiles = torch.cat([low.cpu(), enhanced.cpu(), high.cpu(), residual.cpu()], dim=0)
                vutils.save_image(
                    tiles,
                    comparison_dir / f"{Path(image_name).stem}.png",
                    nrow=4,
                    normalize=False,
                )

    means = {key: float(np.mean(values)) for key, values in results.items()}
    return means, per_image


def main():
    parser = argparse.ArgumentParser(description="Evaluate LumaGuard-Net")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--low_dir", type=str, required=True, help="Directory of low-light images")
    parser.add_argument("--high_dir", type=str, default=None, help="Directory of reference images (if available)")
    parser.add_argument("--output_csv", type=str, default="experiments/evaluation_results.csv", help="CSV output")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'")
    parser.add_argument("--comparison_dir", type=str, default=None,
                        help="Optional directory for input/output/reference/residual comparison images")
    args = parser.parse_args()

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    print(f"[Evaluation] Running on device: {dev}")

    model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=dev)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
            config = ckpt.get("config", None)
            model_type = ckpt.get("model_type", None)
        else:
            state_dict = ckpt
            config = None
            model_type = None

        is_snr = (model_type == "snr_internal_reference") or any(
            "reference_matcher" in k or "feature_transfer" in k for k in state_dict.keys()
        )
        if is_snr:
            model = build_snr_internal_reference_net(config).to(dev)
        else:
            model = build_lumaguard_net(config).to(dev)

        model.load_state_dict(state_dict)
        print(f"[Evaluation] Loaded checkpoint: {args.checkpoint} ({type(model).__name__})")
    else:
        model = LumaGuardNet().to(dev)


    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.high_dir and os.path.exists(args.high_dir):
        # Paired evaluation
        dataset = PairedEvaluationDataset(args.low_dir, args.high_dir)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(dev))
        metrics, per_image = evaluate_paired(
            model, loader, evaluator, dev, comparison_dir=args.comparison_dir
        )
        print(f"[Paired Metrics] {len(per_image)} images | PSNR: {metrics['psnr']:.2f} dB, "
              f"SSIM: {metrics['ssim']:.4f}")
        per_image_csv = out_csv.with_name(out_csv.stem + "_per_image.csv")
        with per_image_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["image", "psnr", "ssim", "mse", "mae"],
                extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(per_image)
        print(f"[Evaluation] Per-image metrics saved to {per_image_csv}")
    else:
        print("[Notice] Reference images not provided. Evaluating no-reference statistical metrics.")
        # Unpaired evaluation
        from datasets.self_supervised_dataset import SelfSupervisedLowLightDataset
        dataset = SelfSupervisedLowLightDataset(args.low_dir, crop_size=None, augment=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        entropies, contrasts = [], []
        for x in tqdm(loader, desc="Evaluating Unpaired"):
            x = x.to(dev)
            with torch.no_grad():
                enh = model(x)
                m = evaluate_no_reference_metrics(enh)
                entropies.append(m["entropy"])
                contrasts.append(m["rms_contrast"])
        metrics = {
            "entropy": float(np.mean(entropies)),
            "rms_contrast": float(np.mean(contrasts)),
        }
        per_image = []
        print(f"[Unpaired Metrics] Entropy: {metrics['entropy']:.2f}, RMS Contrast: {metrics['rms_contrast']:.4f}")

    # Stable schema across paired and unpaired evaluations.
    result_fields = [
        "checkpoint", "low_dir", "high_dir", "num_images",
        "psnr", "ssim", "mse", "mae", "entropy", "rms_contrast",
    ]
    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    with out_csv.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result_fields)
        if not file_exists:
            writer.writeheader()
        row = {key: "" for key in result_fields}
        row.update({
            "checkpoint": args.checkpoint or "None",
            "low_dir": args.low_dir,
            "high_dir": args.high_dir or "",
            "num_images": len(per_image) if args.high_dir else len(dataset),
            **metrics,
        })
        writer.writerow(row)
    print(f"[Evaluation] Results saved to {out_csv}")


if __name__ == "__main__":
    main()
