"""
SNR-Guided Internal Reference Transfer Ablation Suite.
Runs controlled ablations as specified in research protocol:
1. Baseline LumaGuard-Net
2. New model without reference transfer (w/o Ref Transfer)
3. Full new model with reliability-guided reference transfer (SNRInternalReferenceNet)
4. Full model without pseudo-pair reconstruction (w/o L_pseudo)
5. Full model without structure loss (w/o L_gradient)
6. Full model without noise-aware consistency (w/o L_consistency)

Outputs:
- experiments/snr_ablation_matrix.csv with parameter counts and evaluation results.
"""

import copy
import os
import sys
from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.lumaguard_net import LumaGuardNet
from models.snr_internal_reference_net import SNRInternalReferenceNet
from losses.snr_guided_losses import SNRGuidedTotalLoss
from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from metrics.reference_metrics import ReferenceMetricsEvaluator


def evaluate_model_paired(model, val_loader, evaluator, device):
    model.eval()
    psnr_list, ssim_list = [], []

    for low, high, _ in val_loader:
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
            psnr_list.append(m["psnr"])
            ssim_list.append(m["ssim"])

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def main():
    device = torch.device("cpu")
    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))

    eval_low_dir = "data/lol_dataset/eval15/low"
    eval_high_dir = "data/lol_dataset/eval15/high"
    val_loader = None
    if os.path.exists(eval_low_dir) and os.path.exists(eval_high_dir):
        val_dataset = PairedEvaluationDataset(eval_low_dir, eval_high_dir)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    ablation_definitions = [
        {
            "id": 1,
            "name": "1. Existing baseline (LumaGuard-Net)",
            "model_cls": LumaGuardNet,
            "model_kwargs": {},
            "checkpoint": "experiments/lol_v1_10epochs/best_checkpoint.pth",
            "loss_weights": {"w_mask": 0.5, "w_exp": 8.0, "w_smooth": 15.0, "w_color": 3.0, "w_struct": 0.1},
            "description": "Restored baseline with fixed exposure and masked consistency",
        },
        {
            "id": 2,
            "name": "2. New model w/o reference transfer",
            "model_cls": SNRInternalReferenceNet,
            "model_kwargs": {"use_reference_transfer": False},
            "checkpoint": None,
            "loss_weights": {"w_pseudo": 1.0, "w_gradient": 0.5, "w_reference": 0.0, "w_exposure": 4.0, "w_color": 2.0, "w_smooth": 10.0},
            "description": "SNR model with internal reference transfer disabled (local Retinex only)",
        },
        {
            "id": 3,
            "name": "3. Full model with reliability-guided transfer",
            "model_cls": SNRInternalReferenceNet,
            "model_kwargs": {"use_reference_transfer": True},
            "checkpoint": "experiments/snr_reference_lol_v1/best_checkpoint.pth",
            "loss_weights": {"w_pseudo": 1.0, "w_gradient": 0.5, "w_reference": 0.1, "w_exposure": 4.0, "w_color": 2.0, "w_smooth": 10.0},
            "description": "Full proposed architecture with confidence-gated structural transfer",
        },
        {
            "id": 4,
            "name": "4. Full model w/o pseudo-pair reconstruction",
            "model_cls": SNRInternalReferenceNet,
            "model_kwargs": {"use_reference_transfer": True},
            "checkpoint": None,
            "loss_weights": {"w_pseudo": 0.0, "w_gradient": 0.5, "w_reference": 0.1, "w_exposure": 4.0, "w_color": 2.0, "w_smooth": 10.0},
            "description": "Ablates synthetic degraded pseudo-pair reconstruction loss L_pseudo",
        },
        {
            "id": 5,
            "name": "5. Full model w/o structure loss",
            "model_cls": SNRInternalReferenceNet,
            "model_kwargs": {"use_reference_transfer": True},
            "checkpoint": None,
            "loss_weights": {"w_pseudo": 1.0, "w_gradient": 0.0, "w_reference": 0.1, "w_exposure": 4.0, "w_color": 2.0, "w_smooth": 10.0},
            "description": "Ablates reliability-weighted gradient structure loss L_gradient",
        },
        {
            "id": 6,
            "name": "6. Full model w/o noise-aware consistency",
            "model_cls": SNRInternalReferenceNet,
            "model_kwargs": {"use_reference_transfer": True},
            "checkpoint": None,
            "loss_weights": {"w_pseudo": 1.0, "w_gradient": 0.5, "w_reference": 0.1, "w_exposure": 4.0, "w_color": 2.0, "w_smooth": 10.0, "w_consistency": 0.0},
            "description": "Ablates augmentation consistency term L_consistency",
        },
    ]

    print("=" * 105)
    print(f"{'Variant':<48} | {'Params':<8} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Status'}")
    print("=" * 105)

    rows = []
    for item in ablation_definitions:
        cls = item["model_cls"]
        kwargs = item["model_kwargs"]
        model = cls(**kwargs).to(device)
        params = model.count_parameters()

        psnr_val = None
        ssim_val = None
        status = "Configured"

        ckpt_path = item["checkpoint"]
        if ckpt_path and os.path.exists(ckpt_path) and val_loader is not None:
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            try:
                model.load_state_dict(state_dict)
                psnr, ssim = evaluate_model_paired(model, val_loader, evaluator, device)
                psnr_val = f"{psnr:.2f}"
                ssim_val = f"{ssim:.4f}"
                status = "Evaluated"
            except Exception as e:
                status = f"Load error: {e}"

        row = {
            "id": item["id"],
            "variant": item["name"],
            "params": params,
            "psnr": psnr_val if psnr_val else "N/A",
            "ssim": ssim_val if ssim_val else "N/A",
            "status": status,
            "description": item["description"],
        }
        rows.append(row)
        print(f"{item['name']:<48} | {params:<8,} | {row['psnr']:<10} | {row['ssim']:<8} | {status}")

    print("=" * 105)

    out_csv = Path("experiments/snr_ablation_matrix.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "variant", "params", "psnr", "ssim", "status", "description"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Done] Saved SNR ablation matrix to: {out_csv}")


if __name__ == "__main__":
    main()
