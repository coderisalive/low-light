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


def evaluate_paired(model, val_loader, evaluator, device):
    model.eval()
    results = {"psnr": [], "ssim": [], "mse": [], "mae": []}

    for low, high, name in tqdm(val_loader, desc="Evaluating Paired"):
        low = low.to(device)
        high = high.to(device)

        with torch.no_grad():
            b, c, h, w = low.shape
            pad_w = (16 - (w % 16)) % 16
            pad_h = (16 - (h % 16)) % 16
            if pad_w > 0 or pad_h > 0:
                low_pad = torch.nn.functional.pad(low, (0, pad_w, 0, pad_h), mode="reflect")
                enhanced = model(low_pad)[:, :, :h, :w]
            else:
                enhanced = model(low)
            m = evaluator.evaluate(enhanced, high)
            for k in results:
                results[k].append(m[k])

    return {k: float(np.mean(v)) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate LumaGuard-Net")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--low_dir", type=str, required=True, help="Directory of low-light images")
    parser.add_argument("--high_dir", type=str, default=None, help="Directory of reference images (if available)")
    parser.add_argument("--output_csv", type=str, default="experiments/evaluation_results.csv", help="CSV output")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or 'cuda'")
    args = parser.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu")
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
        metrics = evaluate_paired(model, loader, evaluator, dev)
        print(f"[Paired Metrics] PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
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
        print(f"[Unpaired Metrics] Entropy: {metrics['entropy']:.2f}, RMS Contrast: {metrics['rms_contrast']:.4f}")

    # Export to CSV
    file_exists = out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "low_dir", *metrics.keys()])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"checkpoint": args.checkpoint or "None", "low_dir": args.low_dir, **metrics})
    print(f"[Evaluation] Results saved to {out_csv}")


if __name__ == "__main__":
    main()
