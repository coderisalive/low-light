"""
Training Pipeline for SNR-Guided Internal Reference Transfer Network.
Self-Supervised Low-Light Image Enhancement.

Features:
- Self-supervised optimization on unpaired low-light images (zero paired ground-truth access during training).
- Pseudo-pair synthetic degradation supervision on reliable patches.
- Reliability-weighted gradient structure loss.
- Confidence-gated reference feature transfer.
- Adaptive exposure control and edge-aware illumination smoothness.
- Saves checkpoints and diagnostic visual maps to a dedicated experiment folder.
"""

import argparse
import os
import sys
from pathlib import Path
import random

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

from models.snr_internal_reference_net import (
    SNRInternalReferenceNet,
    build_snr_internal_reference_net,
)
from losses.snr_guided_losses import SNRGuidedTotalLoss
from datasets.self_supervised_dataset import SelfSupervisedLowLightDataset
from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from metrics.reference_metrics import ReferenceMetricsEvaluator


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: SNRGuidedTotalLoss,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss_accum = 0.0
    count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        x_low = batch.to(device)

        optimizer.zero_grad()
        loss, loss_dict = loss_fn(model, x_low)
        loss.backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss_accum += loss.item()
        count += 1
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            psd=f"{loss_dict['loss_pseudo']:.3f}",
            exp=f"{loss_dict['loss_exposure']:.3f}",
            sm=f"{loss_dict['loss_smooth']:.3f}",
        )

    return total_loss_accum / max(count, 1)


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    evaluator: ReferenceMetricsEvaluator,
    device: torch.device,
) -> dict:
    model.eval()
    psnr_list, ssim_list = [], []

    for batch in val_loader:
        low, high, _ = batch
        low = low.to(device)
        high = high.to(device)

        # Pad to multiple of 16 for patch grid
        b, c, h, w = low.shape
        pad_w = (16 - (w % 16)) % 16
        pad_h = (16 - (h % 16)) % 16
        if pad_w > 0 or pad_h > 0:
            low_pad = torch.nn.functional.pad(low, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            low_pad = low

        diag = model.forward_diagnostic(low_pad)
        enhanced = diag["enhanced"][:, :, :h, :w]

        metrics = evaluator.evaluate(enhanced, high)
        psnr_list.append(metrics["psnr"])
        ssim_list.append(metrics["ssim"])

    return {
        "val_psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "val_ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train SNR-Guided Internal Reference Transfer Network"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/snr_reference_smoke.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    # Device selection
    dev_str = config["training"].get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    print(f"[SNR-Guided Net] Training on device: {device}")

    # Build model and loss
    model = build_snr_internal_reference_net(config).to(device)
    loss_fn = SNRGuidedTotalLoss(config).to(device)

    print(f"[SNR-Guided Net] Trainable parameters: {model.count_parameters():,}")

    # Datasets
    data_cfg = config.get("data", {})
    train_dir = data_cfg.get("train_dir", "")
    crop_size = config["training"].get("crop_size", 160)
    batch_size = config["training"].get("batch_size", 4)

    train_dataset = SelfSupervisedLowLightDataset(
        train_dir, crop_size=crop_size, augment=True
    )
    if len(train_dataset) == 0:
        print(f"[Error] Train dataset at '{train_dir}' is empty or does not exist!")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Optional paired validation
    val_low = data_cfg.get("val_low_dir")
    val_high = data_cfg.get("val_high_dir")
    val_loader = None
    if val_low and val_high and os.path.exists(val_low) and os.path.exists(val_high):
        val_dataset = PairedEvaluationDataset(val_low, val_high)
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            print(
                f"[SNR-Guided Net] Validation set loaded with {len(val_dataset)} paired samples."
            )

    # Optimizer & Scheduler
    lr = float(config["training"].get("learning_rate", 4e-4))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"].get("scheduler_step_size", 4),
        gamma=config["training"].get("scheduler_gamma", 0.5),
    )

    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))

    # Output experiment folder
    exp_dir = Path("experiments") / config.get(
        "experiment_name", "snr_reference_experiment"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = exp_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    start_epoch = 1
    best_loss = float("inf")
    best_psnr = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"[SNR-Guided Net] Resumed from checkpoint {args.resume} at epoch {start_epoch}")

    epochs = config["training"].get("epochs", 10)
    history = []

    print(f"[SNR-Guided Net] Starting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        scheduler.step()

        log_str = f"Epoch {epoch:03d}/{epochs:03d} - Train Loss: {train_loss:.4f}"

        # Evaluate on validation split if available
        val_metrics = {}
        if val_loader is not None and epoch % config["training"].get("eval_interval", 1) == 0:
            val_metrics = evaluate_validation(model, val_loader, evaluator, device)
            log_str += f" | Val PSNR: {val_metrics['val_psnr']:.2f} dB, SSIM: {val_metrics['val_ssim']:.4f}"

        print(log_str)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        # Save checkpoint
        is_best = False
        if val_loader is not None:
            if val_metrics.get("val_psnr", 0.0) > best_psnr:
                best_psnr = val_metrics["val_psnr"]
                is_best = True
        else:
            if train_loss < best_loss:
                best_loss = train_loss
                is_best = True

        state = {
            "epoch": epoch,
            "model_type": "snr_internal_reference",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": train_loss,
            "best_loss": best_loss,
            "best_psnr": best_psnr,
            "config": config,
        }

        torch.save(state, exp_dir / "latest_checkpoint.pth")
        if is_best:
            torch.save(state, exp_dir / "best_checkpoint.pth")

        # Save diagnostic visualization sample
        if epoch % config["training"].get("save_interval", 1) == 0 or epoch == epochs:
            model.eval()
            sample_batch = next(iter(train_loader)).to(device)[:2]
            with torch.no_grad():
                diag = model.forward_diagnostic(sample_batch)
                sample_vis = torch.cat(
                    [
                        sample_batch,
                        diag["enhanced"],
                        diag["reliability_map"].repeat(1, 3, 1, 1),
                        diag["illumination"].repeat(1, 3, 1, 1),
                        diag["match_confidence"].repeat(1, 3, 1, 1),
                    ],
                    dim=0,
                )
                vutils.save_image(
                    sample_vis,
                    samples_dir / f"epoch_{epoch:03d}.png",
                    nrow=2,
                    normalize=False,
                )

    print(f"[SNR-Guided Net] Training complete. Artifacts saved to: {exp_dir}")


if __name__ == "__main__":
    main()
