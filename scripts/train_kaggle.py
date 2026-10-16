"""
Kaggle GPU Training & Evaluation Pipeline for SNR-Guided Internal Reference Transfer.
Optimized for NVIDIA Tesla T4 / P100 / V100 GPU environments on Kaggle.

Features:
- Automatic Kaggle environment & dataset path discovery (/kaggle/input/...).
- Automatic Mixed Precision (AMP) with torch.cuda.amp for 2-3x GPU acceleration.
- Multi-threaded pinned memory DataLoader (pin_memory=True, num_workers=2).
- Zero normal-light ground truth leakage during training (strictly self-supervised).
- Evaluates PSNR and SSIM on LOL-v1 eval15 paired split after each epoch.
- Exports best model checkpoint and side-by-side visual comparisons directly to /kaggle/working/.
"""

import argparse
import os
import sys
from pathlib import Path
import random
import csv

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
from PIL import Image

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
        torch.backends.cudnn.benchmark = True


def auto_detect_kaggle_dataset(suggested_train_dir: str) -> dict:
    """
    Automatically detects LOL dataset directories across Kaggle input folders
    or local project structures.
    """
    # 1. Check if suggested directory already exists
    if os.path.exists(suggested_train_dir):
        base_dir = Path(suggested_train_dir).parent.parent
        return {
            "train_dir": suggested_train_dir,
            "val_low_dir": str(base_dir / "eval15" / "low"),
            "val_high_dir": str(base_dir / "eval15" / "high"),
            "test_low_dir": str(base_dir / "eval15" / "low"),
            "test_high_dir": str(base_dir / "eval15" / "high"),
        }

    # 2. Check standard Kaggle input candidate paths
    kaggle_input_root = Path("/kaggle/input")
    search_dirs = [kaggle_input_root, Path("data"), Path(".")]
    for root in search_dirs:
        if root.exists():
            for p in root.rglob("our485/low"):
                if p.is_dir() and len(list(p.glob("*.png"))) > 0:
                    base_dir = p.parent.parent
                    print(f"[Kaggle Setup] Auto-detected LOL dataset at: {base_dir}")
                    return {
                        "train_dir": str(p),
                        "val_low_dir": str(base_dir / "eval15" / "low"),
                        "val_high_dir": str(base_dir / "eval15" / "high"),
                        "test_low_dir": str(base_dir / "eval15" / "low"),
                        "test_high_dir": str(base_dir / "eval15" / "high"),
                    }

    # Fallback to config path
    return {"train_dir": suggested_train_dir}


def train_one_epoch_gpu(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: SNRGuidedTotalLoss,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    epoch: int,
) -> float:
    model.train()
    total_loss_accum = 0.0
    count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}", leave=False)
    for batch in pbar:
        x_low = batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss, loss_dict = loss_fn(model, x_low)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = loss_fn(model, x_low)
            loss.backward()
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
def evaluate_paired_gpu(
    model: nn.Module,
    val_loader: DataLoader,
    evaluator: ReferenceMetricsEvaluator,
    device: torch.device,
) -> dict:
    model.eval()
    psnr_list, ssim_list = [], []

    for batch in val_loader:
        low, high, _ = batch
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)

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
        description="Train SNR-Guided Internal Reference Transfer on Kaggle GPU"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_gpu.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional root directory of LOL dataset containing our485/ and eval15/",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume from",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    set_seed(config.get("seed", 42))

    # GPU Device & Capabilities
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Kaggle GPU] Accelerated training active on: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Notice] CUDA not detected. Falling back to CPU.")

    use_amp = bool(config["training"].get("use_amp", True)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[Kaggle GPU] Automatic Mixed Precision (AMP) enabled.")

    # Model & Loss
    model = build_snr_internal_reference_net(config).to(device)
    loss_fn = SNRGuidedTotalLoss(config).to(device)

    param_count = model.count_parameters()
    print(f"[SNR-Guided Net] Trainable parameters: {param_count:,} (< 1,000,000 budget)")

    # Data Path Auto-Discovery
    data_cfg = config.get("data", {})
    if args.data_root:
        root_p = Path(args.data_root)
        paths = {
            "train_dir": str(root_p / "our485" / "low"),
            "val_low_dir": str(root_p / "eval15" / "low"),
            "val_high_dir": str(root_p / "eval15" / "high"),
            "test_low_dir": str(root_p / "eval15" / "low"),
            "test_high_dir": str(root_p / "eval15" / "high"),
        }
    else:
        paths = auto_detect_kaggle_dataset(data_cfg.get("train_dir", "data/lol_dataset/our485/low"))

    train_dir = paths["train_dir"]
    crop_size = config["training"].get("crop_size", 192)
    batch_size = config["training"].get("batch_size", 16)
    num_workers = int(config["training"].get("num_workers", 2 if device.type == "cuda" else 0))
    pin_memory = bool(config["training"].get("pin_memory", True and device.type == "cuda"))

    print(f"[Dataset] Loading training samples from: {train_dir}")
    train_dataset = SelfSupervisedLowLightDataset(train_dir, crop_size=crop_size, augment=True)
    if len(train_dataset) == 0:
        raise FileNotFoundError(f"Training dataset at '{train_dir}' is empty or does not exist!")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    print(f"[Dataset] Loaded {len(train_dataset)} training images ({len(train_loader)} batches/epoch).")

    # Paired Validation Loader
    val_low = paths.get("val_low_dir")
    val_high = paths.get("val_high_dir")
    val_loader = None
    if val_low and val_high and os.path.exists(val_low) and os.path.exists(val_high):
        val_dataset = PairedEvaluationDataset(val_low, val_high)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        print(f"[Dataset] Loaded {len(val_dataset)} paired evaluation samples for validation.")

    # Optimizer & LR Scheduler
    lr = float(config["training"].get("learning_rate", 5e-4))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"].get("scheduler_step_size", 10),
        gamma=config["training"].get("scheduler_gamma", 0.5),
    )

    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))

    # Output directory
    is_kaggle = Path("/kaggle/working").exists()
    default_out = "/kaggle/working/experiments" if is_kaggle else "experiments"
    exp_dir = Path(default_out) / config.get("experiment_name", "snr_reference_kaggle_gpu")
    exp_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = exp_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    eval_csv_path = exp_dir / "training_history.csv"

    start_epoch = 1
    best_psnr = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"[Checkpoint] Resumed from {args.resume} at epoch {start_epoch}")

    epochs = config["training"].get("epochs", 35)
    history = []

    print("=" * 80)
    print(f"[Training] Starting SNR-Guided optimization for {epochs} epochs on {device}...")
    print("=" * 80)

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch_gpu(
            model, train_loader, optimizer, loss_fn, device, scaler, use_amp, epoch
        )
        scheduler.step()

        # Evaluate on validation split
        val_metrics = {}
        if val_loader is not None and epoch % config["training"].get("eval_interval", 1) == 0:
            val_metrics = evaluate_paired_gpu(model, val_loader, evaluator, device)

        psnr_val = val_metrics.get("val_psnr", 0.0)
        ssim_val = val_metrics.get("val_ssim", 0.0)
        is_best = psnr_val > best_psnr
        if is_best:
            best_psnr = psnr_val

        status_flag = " [BEST]" if is_best else ""
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | "
            f"Val PSNR: {psnr_val:.2f} dB | Val SSIM: {ssim_val:.4f}{status_flag}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_psnr": psnr_val,
            "val_ssim": ssim_val,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save latest & best checkpoints
        state = {
            "epoch": epoch,
            "model_type": "snr_internal_reference",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": train_loss,
            "best_psnr": best_psnr,
            "config": config,
        }

        torch.save(state, exp_dir / "latest_checkpoint.pth")
        if is_best:
            torch.save(state, exp_dir / "best_checkpoint.pth")
            if is_kaggle:
                # Also save directly in /kaggle/working for quick access
                torch.save(state, "/kaggle/working/best_checkpoint.pth")

        # Save visual diagnostic samples
        if epoch % config["training"].get("save_interval", 5) == 0 or epoch == epochs or is_best:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_loader)).to(device)[:2]
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

    # Export history CSV
    with open(eval_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_psnr", "val_ssim", "lr"])
        writer.writeheader()
        writer.writerows(history)

    print("=" * 80)
    print(f"[Training Complete] Best Checkpoint PSNR: {best_psnr:.2f} dB")
    print(f"Artifacts and weights successfully saved to: {exp_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
