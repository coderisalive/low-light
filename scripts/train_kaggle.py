"""
Stable Kaggle GPU training for SNR-guided LumaGuard-Net.

Key safeguards:
- Self-supervised optimization on low-light images only.
- Deterministic hold-out split from our485 for paired validation metrics.
- eval15 is not used for epoch-by-epoch checkpoint selection.
- AdamW + ReduceLROnPlateau, gradient clipping, AMP, early stopping.
- Resume-safe optimizer/scheduler/scaler/history state and separate best/latest checkpoints.
- Per-image validation CSV and input/output/ground-truth/residual comparison exports.
"""
import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torchvision.utils as vutils

from models.snr_internal_reference_net import build_snr_internal_reference_net
from losses.snr_guided_losses import SNRGuidedTotalLoss
from datasets.self_supervised_dataset import SelfSupervisedLowLightDataset
from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from metrics.reference_metrics import ReferenceMetricsEvaluator


VALIDATION_PROTOCOL = "heldout_our485_pairs_v1"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Favor repeatability over autotuner-based speed variation.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def auto_detect_kaggle_dataset(suggested_train_dir: str) -> dict:
    """Detect a LOL root while keeping eval15 out of epoch-by-epoch validation."""
    suggested = Path(suggested_train_dir)
    if suggested.is_dir():
        root = suggested.parent.parent
        return {
            "train_dir": str(suggested),
            "train_high_dir": str(root / "our485" / "high"),
            "test_low_dir": str(root / "eval15" / "low"),
            "test_high_dir": str(root / "eval15" / "high"),
        }

    search_dirs = [Path("/kaggle/input"), Path("data"), Path(".")]
    for search_root in search_dirs:
        if not search_root.exists():
            continue
        for low_dir in search_root.rglob("our485/low"):
            if low_dir.is_dir() and any(low_dir.glob("*")):
                root = low_dir.parent.parent
                print(f"[Dataset] Auto-detected LOL root: {root}")
                return {
                    "train_dir": str(low_dir),
                    "train_high_dir": str(root / "our485" / "high"),
                    "test_low_dir": str(root / "eval15" / "low"),
                    "test_high_dir": str(root / "eval15" / "high"),
                }

    # Last-resort paths derived from the configured training directory.
    return {
        "train_dir": str(suggested),
        "train_high_dir": str(suggested.parent / "high"),
    }


def train_one_epoch_gpu(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: SNRGuidedTotalLoss,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    totals: Dict[str, float] = {}
    batch_count = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}", leave=False)

    for x_low in pbar:
        x_low = x_low.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
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

        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")

        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batch_count += 1
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            pseudo=f"{loss_dict.get('loss_pseudo', 0.0):.3f}",
            grad=f"{loss_dict.get('loss_gradient', 0.0):.3f}",
            exp=f"{loss_dict.get('loss_exposure', 0.0):.3f}",
            smooth=f"{loss_dict.get('loss_smooth', 0.0):.3f}",
        )

    if batch_count == 0:
        raise RuntimeError("Training DataLoader produced zero batches. Check dataset size and batch settings.")

    mean_components = {key: value / batch_count for key, value in totals.items()}
    return mean_components.get("total_loss", 0.0), mean_components


@torch.no_grad()
def evaluate_paired_gpu(
    model: nn.Module,
    val_loader: DataLoader,
    evaluator: ReferenceMetricsEvaluator,
    device: torch.device,
) -> dict:
    model.eval()
    per_image = []

    for low, high, name in tqdm(val_loader, desc="Held-out validation", leave=False):
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)

        _, _, height, width = low.shape
        pad_w = (16 - width % 16) % 16
        pad_h = (16 - height % 16) % 16
        if pad_w or pad_h:
            low_pad = F.pad(low, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            low_pad = low

        output = model(low_pad)[:, :, :height, :width].clamp(0.0, 1.0)
        metrics = evaluator.evaluate(output, high.clamp(0.0, 1.0))
        image_name = name[0] if isinstance(name, (tuple, list)) else str(name)
        per_image.append({"image": image_name, **metrics})

    if not per_image:
        raise RuntimeError("Validation produced no paired images.")

    keys = ("psnr", "ssim", "mse", "mae")
    means = {key: float(np.mean([row[key] for row in per_image])) for key in keys}
    means["per_image"] = per_image
    return means


@torch.no_grad()
def save_validation_comparisons(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    max_images: int = 4,
) -> None:
    """Save [low | enhanced | reference | amplified absolute residual] tiles."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (low, high, name) in enumerate(val_loader):
        if index >= max_images:
            break
        low = low.to(device)
        high = high.to(device)
        _, _, height, width = low.shape
        pad_w = (16 - width % 16) % 16
        pad_h = (16 - height % 16) % 16
        low_pad = F.pad(low, (0, pad_w, 0, pad_h), mode="reflect") if pad_w or pad_h else low
        enhanced = model(low_pad)[:, :, :height, :width].clamp(0.0, 1.0)
        residual = ((enhanced - high).abs() * 4.0).clamp(0.0, 1.0)
        tiles = torch.cat([low.cpu(), enhanced.cpu(), high.cpu(), residual.cpu()], dim=0)
        image_name = name[0] if isinstance(name, (tuple, list)) else str(name)
        safe_name = Path(image_name).stem
        vutils.save_image(tiles, output_dir / f"epoch_{epoch:03d}_{safe_name}.png", nrow=4, normalize=False)


def make_deterministic_split(total_items: int, validation_fraction: float, seed: int):
    """Return disjoint, deterministic train/validation indices."""
    if total_items < 2:
        raise ValueError("At least two items are required for a train/validation split.")
    if not 0.05 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0.05, 0.5).")
    indices = list(range(total_items))
    random.Random(seed).shuffle(indices)
    val_count = max(1, min(total_items - 1, int(round(total_items * validation_fraction))))
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])
    return train_indices, val_indices


def _load_history_from_csv(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable SNR-guided self-supervised training")
    parser.add_argument("--config", type=str, default="configs/kaggle_gpu.yaml")
    parser.add_argument("--data_root", type=str, default=None,
                        help="LOL root containing our485/low, our485/high and eval15/")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    training_cfg = config.setdefault("training", {})
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    use_amp = bool(training_cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model = build_snr_internal_reference_net(config).to(device)
    loss_fn = SNRGuidedTotalLoss(config).to(device)
    print(f"[Model] {model.count_parameters():,} trainable parameters")
    print(f"[Optimizer] AdamW; base LR={training_cfg.get('learning_rate', 2e-4):.2e}")

    data_cfg = config.setdefault("data", {})
    if args.data_root:
        root = Path(args.data_root)
        paths = {
            "train_dir": str(root / "our485" / "low"),
            "train_high_dir": str(root / "our485" / "high"),
            "test_low_dir": str(root / "eval15" / "low"),
            "test_high_dir": str(root / "eval15" / "high"),
        }
    else:
        paths = auto_detect_kaggle_dataset(data_cfg.get("train_dir", "data/lol_dataset/our485/low"))
    train_dir = paths["train_dir"]
    train_high_dir = paths.get("train_high_dir", data_cfg.get("train_high_dir", str(Path(train_dir).parent / "high")))

    if not Path(train_dir).is_dir():
        raise FileNotFoundError(f"Training low-light directory not found: {train_dir}")
    if not Path(train_high_dir).is_dir():
        raise FileNotFoundError(
            f"Paired training reference directory not found: {train_high_dir}. "
            "The script uses a held-out our485 subset for validation and deliberately does not "
            "use eval15 for checkpoint selection."
        )

    crop_size = int(training_cfg.get("crop_size", 192))
    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 2 if device.type == "cuda" else 0))
    pin_memory = bool(training_cfg.get("pin_memory", device.type == "cuda")) and device.type == "cuda"

    full_train_dataset = SelfSupervisedLowLightDataset(train_dir, crop_size=crop_size, augment=True)
    total_images = len(full_train_dataset)
    if total_images < 2:
        raise RuntimeError(f"Need at least 2 training images to create a hold-out split; found {total_images}.")

    validation_fraction = float(training_cfg.get("validation_fraction", 0.10))
    train_indices, val_indices = make_deterministic_split(total_images, validation_fraction, seed)
    val_names = [full_train_dataset.image_paths[i].name for i in val_indices]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = PairedEvaluationDataset(train_dir, train_high_dir, filenames=val_names)
    if len(val_dataset) != len(val_indices):
        raise RuntimeError(
            f"Held-out pair mismatch: selected {len(val_indices)} low-light files but found "
            f"{len(val_dataset)} matching low/high pairs. Check filenames in our485/low and our485/high."
        )
    if not len(train_dataset):
        raise RuntimeError("Training split is empty.")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    print(f"[Data] Total our485 images: {total_images}; train: {len(train_dataset)}; "
          f"held-out validation: {len(val_dataset)}. eval15 is reserved for final testing.")

    lr = float(training_cfg.get("learning_rate", 2e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_cfg.get("scheduler_gamma", 0.5)),
        patience=int(training_cfg.get("scheduler_patience", 3)),
        threshold=float(training_cfg.get("lr_min_delta", 0.01)),
        threshold_mode="abs",
        cooldown=1,
        min_lr=float(training_cfg.get("min_learning_rate", 1e-6)),
    )
    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))

    root_output = Path("/kaggle/working/experiments") if Path("/kaggle/working").exists() else Path("experiments")
    experiment_name = str(config.get("experiment_name", "snr_reference_stable_v3"))
    exp_dir = root_output / experiment_name
    resume_path = Path(args.resume).resolve() if args.resume else None
    if exp_dir.exists() and any(exp_dir.iterdir()):
        same_resume_dir = resume_path is not None and resume_path.parent == exp_dir.resolve()
        if not same_resume_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = exp_dir.with_name(f"{exp_dir.name}_run_{stamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = exp_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    history_csv = exp_dir / "training_history.csv"
    per_image_csv = exp_dir / "validation_per_image.csv"

    start_epoch = 1
    best_psnr = float("-inf")
    best_ssim = float("-inf")
    bad_epochs = 0
    history = []

    if args.resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if bool(training_cfg.get("resume_override_lr", True)):
                for group in optimizer.param_groups:
                    group["lr"] = lr
        scheduler_state = checkpoint.get("scheduler")
        # Old checkpoints used StepLR. Do not load its incompatible state into ReduceLROnPlateau.
        if isinstance(scheduler_state, dict) and {"mode", "best", "num_bad_epochs"}.issubset(scheduler_state):
            scheduler.load_state_dict(scheduler_state)
        elif scheduler_state is not None:
            print("[Resume] Old scheduler type detected; starting the new plateau scheduler fresh.")
        if "scaler" in checkpoint and use_amp:
            try:
                scaler.load_state_dict(checkpoint["scaler"])
            except (ValueError, RuntimeError, KeyError) as error:
                print(f"[Resume] AMP scaler state reset: {error}")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if checkpoint.get("validation_protocol") == VALIDATION_PROTOCOL:
            best_psnr = float(checkpoint.get("best_psnr", float("-inf")))
            best_ssim = float(checkpoint.get("best_ssim", float("-inf")))
            bad_epochs = int(checkpoint.get("bad_epochs", 0))
            history = list(checkpoint.get("history", []))
        else:
            print("[Resume] Checkpoint was evaluated under a different validation protocol; "
                  "resetting best metric/patience state but preserving model weights.")
        print(f"[Resume] Loaded {resume_path}; next epoch={start_epoch}")

    epochs = int(training_cfg.get("epochs", 35))
    patience = int(training_cfg.get("early_stopping_patience", 8))
    min_delta = float(training_cfg.get("early_stopping_min_delta", 0.01))
    eval_interval = max(1, int(training_cfg.get("eval_interval", 1)))
    save_interval = max(1, int(training_cfg.get("save_interval", 5)))

    print(f"[Train] {epochs} total epochs; early-stopping patience={patience}; "
          "checkpoint selection by held-out our485 PSNR.")
    stopped_early = False

    for epoch in range(start_epoch, epochs + 1):
        train_loss, component_means = train_one_epoch_gpu(
            model, train_loader, optimizer, loss_fn, device, scaler, use_amp, epoch
        )

        val_metrics = {}
        evaluated_this_epoch = epoch % eval_interval == 0
        if evaluated_this_epoch:
            val_metrics = evaluate_paired_gpu(model, val_loader, evaluator, device)
            val_psnr = float(val_metrics["psnr"])
            val_ssim = float(val_metrics["ssim"])
            is_best = val_psnr > best_psnr + min_delta
            if is_best:
                best_psnr = val_psnr
                best_ssim = val_ssim
                bad_epochs = 0
            else:
                bad_epochs += 1
            scheduler.step(val_psnr)
        else:
            is_best = False

        lr_now = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **component_means,
            "val_psnr": val_metrics.get("psnr", ""),
            "val_ssim": val_metrics.get("ssim", ""),
            "lr": lr_now,
            "best_psnr": best_psnr if np.isfinite(best_psnr) else "",
            "best_ssim": best_ssim if np.isfinite(best_ssim) else "",
            "bad_epochs": bad_epochs,
        }
        history.append(row)

        if evaluated_this_epoch:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | train={train_loss:.4f} | "
                f"PSNR={val_metrics['psnr']:.3f} dB | SSIM={val_metrics['ssim']:.4f} | "
                f"LR={lr_now:.2e} | patience={bad_epochs}/{patience}"
                + (" [BEST]" if is_best else "")
            )
            with per_image_csv.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["epoch", "image", "psnr", "ssim", "mse", "mae"])
                if stream.tell() == 0:
                    writer.writeheader()
                for item in val_metrics["per_image"]:
                    writer.writerow({"epoch": epoch, **item})

        state = {
            "epoch": epoch,
            "model_type": "snr_internal_reference",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else {},
            "train_loss": train_loss,
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "bad_epochs": bad_epochs,
            "history": history,
            "validation_protocol": VALIDATION_PROTOCOL,
            "validation_split_seed": seed,
            "validation_fraction": validation_fraction,
            "config": config,
        }
        torch.save(state, exp_dir / "latest_checkpoint.pth")
        if is_best:
            torch.save(state, exp_dir / "best_checkpoint.pth")
            if Path("/kaggle/working").exists():
                torch.save(state, Path("/kaggle/working") / f"{exp_dir.name}_best_checkpoint.pth")
            save_validation_comparisons(model, val_loader, device, comparison_dir, epoch)

        if epoch % save_interval == 0 and not is_best:
            save_validation_comparisons(model, val_loader, device, comparison_dir, epoch)

        if evaluated_this_epoch and not is_best and bad_epochs >= patience:
            print(f"[EarlyStopping] No validation PSNR improvement > {min_delta:.3f} dB "
                  f"for {bad_epochs} consecutive evaluations. Stopping at epoch {epoch}.")
            stopped_early = True
            break

    fieldnames = [
        "epoch", "train_loss", "total_loss", "loss_pseudo", "loss_gradient",
        "loss_reference", "loss_exposure", "loss_color", "loss_smooth",
        "loss_consistency", "val_psnr", "val_ssim", "lr", "best_psnr",
        "best_ssim", "bad_epochs",
    ]
    with history_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)

    print("=" * 80)
    print(f"[Complete] Best held-out validation PSNR: {best_psnr:.3f} dB; SSIM at that epoch: {best_ssim:.4f}")
    print(f"[Complete] Early stopped: {stopped_early}; artifacts: {exp_dir}")
    print(f"[Complete] Per-image validation metrics: {per_image_csv}")
    print(f"[Complete] Comparisons: {comparison_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
