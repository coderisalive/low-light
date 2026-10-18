"""Train a new lightweight RGB U-Net on paired LOL-v1 images.

Training and checkpoint selection use only a deterministic split from our485.
The eval15 directory is not loaded by this script.
"""
import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from losses.paired_rgb_restoration import paired_rgb_restoration_loss
from metrics.reference_metrics import ReferenceMetricsEvaluator
from models.paired_rgb_restoration_net import PairedRGBRestorationNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_rgb8(x: torch.Tensor) -> np.ndarray:
    return (x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def save_labeled_montage(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    images = [Image.fromarray(array.astype(np.uint8), mode="RGB") for _, array in panels]
    if not images or any(image.size != images[0].size for image in images):
        raise ValueError("Comparison panels must be non-empty and have identical dimensions.")
    panel_w, panel_h = images[0].size
    label_h = 34
    canvas = Image.new("RGB", (panel_w * len(images), panel_h + label_h), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for index, ((label, _), image) in enumerate(zip(panels, images)):
        x = index * panel_w
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        draw.text((x + max(3, (panel_w - tw) // 2), (label_h - th) // 2 - box[1]),
                  label, fill="white", font=font)
        canvas.paste(image, (x, label_h))
        if index:
            draw.line((x, 0, x, panel_h + label_h), fill=(95, 95, 95), width=1)
    canvas.save(path)


def save_comparisons(model, loader, device, out_dir: Path, max_images: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for index, (low, high, names) in enumerate(loader):
            if index >= max_images:
                break
            low = low.to(device)
            high = high.to(device)
            output = model(low).clamp(0, 1)
            name = names[0] if isinstance(names, (tuple, list)) else str(names)
            error = ((output[0] - high[0]).abs() * 4.0).clamp(0, 1)
            save_labeled_montage(
                out_dir / f"{Path(name).stem}.png",
                [
                    ("Low-light Input", tensor_to_rgb8(low[0])),
                    ("RGB U-Net Output", tensor_to_rgb8(output[0])),
                    ("Ground Truth", tensor_to_rgb8(high[0])),
                    ("Residual x4", tensor_to_rgb8(error)),
                ],
            )


@torch.inference_mode()
def evaluate(model, loader, evaluator, device, capture_rows: bool = True):
    model.eval()
    metric_rows = []
    loss_values = []
    component_rows = []

    for low, high, names in tqdm(loader, desc="Validation", leave=False):
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)
        output = model(low).clamp(0, 1)
        loss, terms = paired_rgb_restoration_loss(output, high)
        metrics = evaluator.evaluate(output, high)
        name = names[0] if isinstance(names, (tuple, list)) else str(names)
        loss_values.append(float(loss.item()))
        component_rows.append({key: float(value.item()) for key, value in terms.items()})
        if capture_rows:
            metric_rows.append({"image": name, **metrics, "paired_loss": float(loss.item())})

    if not loss_values:
        raise RuntimeError("Validation loader produced zero images.")

    keys = ("psnr", "ssim", "mse", "mae")
    summary = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in keys
    }
    summary["paired_loss"] = float(np.mean(loss_values))
    for key in ("loss_rgb", "loss_luminance", "loss_chroma", "loss_gradient", "loss_contrast"):
        summary[key] = float(np.mean([row[key] for row in component_rows]))
    return summary, metric_rows


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, base_channels: int):
    torch.save(
        {
            "model_type": "paired_rgb_restoration",
            "model": model.state_dict(),
            "model_config": {"base_channels": base_channels},
            "epoch": epoch,
            "validation_metrics": metrics,
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paired-supervised RGB U-Net.")
    parser.add_argument("--data_root", required=True, help="LOL root containing our485/low and our485/high.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_batches", type=int, default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.crop_size < 32:
        raise ValueError("epochs/batch_size must be positive and crop_size >= 32.")
    if args.lr <= 0 or args.patience < 1 or args.base_channels < 8:
        raise ValueError("lr/patience/base_channels are invalid.")
    if not 0.05 <= args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0.05, 0.5).")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("--max_train_batches must be positive.")

    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    data_root = Path(args.data_root).resolve()
    low_dir = data_root / "our485" / "low"
    high_dir = data_root / "our485" / "high"
    if not low_dir.is_dir() or not high_dir.is_dir():
        raise FileNotFoundError(f"Expected paired folders: {low_dir} and {high_dir}")

    train_all = PairedEvaluationDataset(str(low_dir), str(high_dir), crop_size=args.crop_size)
    val_all = PairedEvaluationDataset(str(low_dir), str(high_dir), crop_size=None)
    if len(train_all) < 2 or len(train_all) != len(val_all):
        raise RuntimeError(f"Matched image count mismatch: train={len(train_all)}, val_source={len(val_all)}.")

    shuffled = list(range(len(train_all)))
    random.Random(args.seed).shuffle(shuffled)
    val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * args.validation_fraction)))
    val_indices = sorted(shuffled[:val_count])
    train_indices = sorted(shuffled[val_count:])
    train_ds, val_ds = Subset(train_all, train_indices), Subset(val_all, val_indices)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=use_amp, drop_last=False, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=use_amp,
    )
    print(f"[Data] our485 pairs={len(train_all)}; train={len(train_ds)}; validation={len(val_ds)}; eval15 not loaded.")

    model = PairedRGBRestorationNet(base_channels=args.base_channels).to(device)
    print(f"[Model] PairedRGBRestorationNet | {model.count_parameters():,} trainable parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )
    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path("/kaggle/working/experiments") if Path("/kaggle/working").exists() else Path("experiments")
    exp_dir = output_root / f"paired_rgb_unet_{stamp}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    compare_dir = exp_dir / "comparisons"
    best_path = exp_dir / "best_checkpoint.pth"
    latest_path = exp_dir / "latest_checkpoint.pth"
    portable_path = output_root.parent / f"{exp_dir.name}_best_checkpoint.pth"
    history_path = exp_dir / "training_history.csv"
    per_image_path = exp_dir / "validation_per_image.csv"

    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sums: Dict[str, float] = {}
        batches = 0
        progress = tqdm(train_loader, desc=f"RGB U-Net {epoch:02d}/{args.epochs}", leave=False)

        for batch_index, (low, high, _) in enumerate(progress, start=1):
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)
            if random.random() < 0.5:
                low, high = torch.flip(low, dims=[3]), torch.flip(high, dims=[3])
            if random.random() < 0.5:
                low, high = torch.flip(low, dims=[2]), torch.flip(high, dims=[2])

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(low)
                loss, terms = paired_rgb_restoration_loss(output, high)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            for key, value in terms.items():
                train_sums[key] = train_sums.get(key, 0.0) + float(value.detach().item())
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", rgb=f"{terms['loss_rgb'].item():.4f}")
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                print(f"[Smoke test] Stopping after {batch_index} batches.")
                break

        if batches == 0:
            raise RuntimeError("Training loader produced zero batches.")
        train_means = {key: value / batches for key, value in train_sums.items()}
        val_metrics, val_rows = evaluate(model, val_loader, evaluator, device)
        scheduler.step(val_metrics["paired_loss"])
        improved = val_metrics["paired_loss"] < best_loss - 1e-5
        if improved:
            best_loss = val_metrics["paired_loss"]
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, val_metrics, args.base_channels)
            shutil.copy2(best_path, portable_path)
            save_comparisons(model, val_loader, device, compare_dir, max_images=8)
            with per_image_path.open("w", newline="", encoding="utf-8") as stream:
                fields = ["image", "psnr", "ssim", "mse", "mae", "paired_loss"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(val_rows)
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "train_loss": train_means.get("loss_total", 0.0),
            "train_rgb": train_means.get("loss_rgb", 0.0),
            "train_chroma": train_means.get("loss_chroma", 0.0),
            "val_paired_loss": val_metrics["paired_loss"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "val_mae": val_metrics["mae"],
            "lr": optimizer.param_groups[0]["lr"],
            "best_epoch": best_epoch,
        }
        history.append(row)
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        save_checkpoint(latest_path, model, optimizer, epoch, val_metrics, args.base_channels)

        status = " [BEST]" if improved else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train={row['train_loss']:.5f} | "
            f"val_obj={val_metrics['paired_loss']:.5f} | PSNR={val_metrics['psnr']:.3f} dB | "
            f"SSIM={val_metrics['ssim']:.4f} | MAE={val_metrics['mae']:.5f} | "
            f"LR={row['lr']:.2e} | best={best_epoch} | patience={bad_epochs}/{args.patience}{status}"
        )
        if args.max_train_batches is not None:
            print("[Smoke test] Finished requested smoke-test epoch.")
            break
        if bad_epochs >= args.patience:
            print(f"[Early stopping] No paired validation objective improvement for {args.patience} epochs.")
            break

    print("=" * 76)
    print(f"[Complete] Best epoch: {best_epoch}")
    print(f"[Complete] Best validation objective: {best_loss:.5f}")
    print(f"[Complete] Best checkpoint: {best_path}")
    print(f"[Complete] Portable checkpoint: {portable_path}")
    print(f"[Complete] History: {history_path}")
    print(f"[Complete] Per-image validation metrics: {per_image_path}")
    print(f"[Complete] Labeled comparisons: {compare_dir}")
    print("[Protocol] eval15 was not loaded.")


if __name__ == "__main__":
    main()
