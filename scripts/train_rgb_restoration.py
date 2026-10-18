"""Train PairedRGBRestorationNet on multiple paired low-light datasets.

Supports:
- LOL-v1 (our485/low and our485/high)
- LOL-v2-Real (Real_captured/Train/Low and Normal)
- LOL-v2-Synthetic (Synthetic/Train/Low and Normal)

Training and checkpoint selection use only deterministic splits carved out of the training pools.
Official test sets (eval15, LOL-v2-Real Test, LOL-v2-Synthetic Test) are strictly isolated.
"""
import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from datasets.multi_dataset import (
    MultiDatasetPairedDataset,
    SinglePairedDataset,
    build_multi_paired_train_val,
)
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


def save_comparisons_multi(model, val_loaders: Dict[str, DataLoader], device, out_dir: Path, max_per_ds: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for ds_name, loader in val_loaders.items():
            for index, (low, high, names, unique_ids) in enumerate(loader):
                if index >= max_per_ds:
                    break
                low = low.to(device)
                high = high.to(device)
                output = model(low).clamp(0, 1)
                name = names[0] if isinstance(names, (tuple, list)) else str(names)
                error = ((output[0] - high[0]).abs() * 4.0).clamp(0, 1)
                save_labeled_montage(
                    out_dir / f"{ds_name}_{Path(name).stem}.png",
                    [
                        (f"[{ds_name}] Low-light", tensor_to_rgb8(low[0])),
                        ("RGB U-Net Output", tensor_to_rgb8(output[0])),
                        ("Ground Truth", tensor_to_rgb8(high[0])),
                        ("Residual x4", tensor_to_rgb8(error)),
                    ],
                )


@torch.inference_mode()
def evaluate_dataset(model, loader: DataLoader, evaluator: ReferenceMetricsEvaluator, device, ds_name: str):
    model.eval()
    metric_rows = []
    loss_values = []
    component_rows = []

    for low, high, names, unique_ids in loader:
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)
        output = model(low).clamp(0, 1)
        loss, terms = paired_rgb_restoration_loss(output, high)
        metrics = evaluator.evaluate(output, high)
        name = names[0] if isinstance(names, (tuple, list)) else str(names)
        uid = unique_ids[0] if isinstance(unique_ids, (tuple, list)) else str(unique_ids)

        loss_values.append(float(loss.item()))
        component_rows.append({key: float(value.item()) for key, value in terms.items()})
        metric_rows.append({
            "dataset": ds_name,
            "unique_id": uid,
            "image": name,
            **metrics,
            "paired_loss": float(loss.item()),
        })

    if not loss_values:
        raise RuntimeError(f"Validation loader for {ds_name} produced zero images.")

    keys = ("psnr", "ssim", "mse", "mae")
    summary = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in keys
    }
    summary["paired_loss"] = float(np.mean(loss_values))
    for key in ("loss_rgb", "loss_luminance", "loss_chroma", "loss_gradient", "loss_contrast"):
        summary[key] = float(np.mean([row[key] for row in component_rows]))
    return summary, metric_rows


def evaluate_all_val(model, val_loaders: Dict[str, DataLoader], evaluator: ReferenceMetricsEvaluator, device):
    all_summaries: Dict[str, Dict[str, float]] = {}
    all_rows: List[Dict[str, Any]] = []

    for ds_name, loader in val_loaders.items():
        summary, rows = evaluate_dataset(model, loader, evaluator, device, ds_name)
        all_summaries[ds_name] = summary
        all_rows.extend(rows)

    # Compute macro-average across enabled datasets
    macro_summary: Dict[str, float] = {}
    for key in ("paired_loss", "psnr", "ssim", "mse", "mae", "loss_rgb", "loss_luminance", "loss_chroma", "loss_gradient", "loss_contrast"):
        macro_summary[key] = float(np.mean([all_summaries[d][key] for d in all_summaries]))

    return macro_summary, all_summaries, all_rows


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, base_channels: int, datasets_trained: list):
    torch.save(
        {
            "model_type": "paired_rgb_restoration",
            "model": model.state_dict(),
            "model_config": {"base_channels": base_channels},
            "epoch": epoch,
            "validation_metrics": metrics,
            "datasets_trained": datasets_trained,
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def parse_weights_arg(w_str: Optional[str]) -> Optional[Dict[str, float]]:
    if not w_str:
        return None
    res = {}
    for part in w_str.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            res[k.strip().lower()] = float(v.strip())
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PairedRGBRestorationNet with multi-dataset support.")
    # Dataset path options
    parser.add_argument("--data_root", default=None, help="Backwards-compatible alias for lol_v1 root.")
    parser.add_argument("--lolv1_root", default=None, help="Root folder for LOL-v1 (containing our485).")
    parser.add_argument("--lolv2_real_root", default=None, help="Root folder for LOL-v2-Real (containing Real_captured).")
    parser.add_argument("--lolv2_synthetic_root", default=None, help="Root folder for LOL-v2-Synthetic (containing Synthetic).")

    # Sampling and hyperparameters
    parser.add_argument("--sampling_mode", choices=["proportional", "balanced", "weighted"], default="balanced",
                        help="Sampling strategy across datasets: 'balanced' (equal representation), 'proportional', or 'weighted'.")
    parser.add_argument("--dataset_weights", default=None, type=str,
                        help="Comma-separated dataset weights for weighted mode, e.g. 'lol_v1:1.0,lol_v2_real:1.5'.")
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
    parser.add_argument("--output_dir", default=None, type=str, help="Destination directory for experiment artifacts.")
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

    # 1. Resolve dataset roots
    roots_to_check: Dict[str, Optional[Path]] = {}
    v1_candidate = args.lolv1_root or args.data_root
    if v1_candidate:
        roots_to_check["lol_v1"] = Path(v1_candidate).resolve()
    if args.lolv2_real_root:
        roots_to_check["lol_v2_real"] = Path(args.lolv2_real_root).resolve()
    if args.lolv2_synthetic_root:
        roots_to_check["lol_v2_synthetic"] = Path(args.lolv2_synthetic_root).resolve()

    # Fallback auto-detection if no flags passed but standard folders exist
    if not roots_to_check:
        candidates = [
            ("lol_v1", Path("data/lol_dataset")),
            ("lol_v1", Path("/kaggle/working/lol_compat")),
            ("lol_v2_real", Path("data/LOL-v2-Real")),
            ("lol_v2_real", Path("data/LOLv2_Real")),
            ("lol_v2_synthetic", Path("data/LOL-v2-Synthetic")),
            ("lol_v2_synthetic", Path("data/LOLv2_Synthetic")),
        ]
        for name, p in candidates:
            if p.is_dir() and name not in roots_to_check:
                roots_to_check[name] = p.resolve()

    if not roots_to_check:
        raise FileNotFoundError(
            "No dataset roots specified. Please provide at least one of --lolv1_root, "
            "--lolv2_real_root, or --lolv2_synthetic_root (or --data_root)."
        )

    print("\n=== Dataset Discovery & Configuration ===")
    for k, v in roots_to_check.items():
        print(f"  {k}: {v}")

    # 2. Build multi-dataset paired training and validation loaders
    train_dataset, val_splits, stats = build_multi_paired_train_val(
        dataset_roots=roots_to_check,
        crop_size=args.crop_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        augment=True,
    )

    print("\n=== Dataset Pair Statistics ===")
    for name, s in stats.items():
        print(f"  [{name.upper()}] Available: {s['total_available']} | Train: {s['train_count']} | Val: {s['val_count']} | Unmatched Low: {s['unmatched_low']} | Unmatched High: {s['unmatched_high']}")
    print(f"  --> TOTAL POOLED TRAIN SAMPLES: {len(train_dataset):,}")

    # Build sampler
    custom_w = parse_weights_arg(args.dataset_weights)
    sample_weights = train_dataset.get_sampling_weights(mode=args.sampling_mode, custom_weights=custom_w)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True) if args.sampling_mode != "proportional" else None

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        drop_last=False,
        generator=generator,
    )

    val_loaders = {
        name: DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=use_amp)
        for name, ds in val_splits.items()
    }

    # 3. Model setup
    model = PairedRGBRestorationNet(base_channels=args.base_channels).to(device)
    print(f"\n[Model] PairedRGBRestorationNet | {model.count_parameters():,} trainable parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )
    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))

    # 4. Experiment directories
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        output_root = Path("/kaggle/working/experiments") if Path("/kaggle/working").exists() else Path("experiments")
        exp_dir = output_root / f"paired_rgb_unet_{stamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = exp_dir / "comparisons"
    best_path = exp_dir / "best_checkpoint.pth"
    latest_path = exp_dir / "latest_checkpoint.pth"
    history_path = exp_dir / "training_history.csv"
    per_image_path = exp_dir / "validation_per_image.csv"
    per_dataset_path = exp_dir / "validation_per_dataset.csv"

    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    datasets_trained_list = list(stats.keys())

    print(f"[Experiment Output] {exp_dir.resolve()}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sums: Dict[str, float] = {}
        batches = 0
        progress = tqdm(train_loader, desc=f"RGB U-Net {epoch:02d}/{args.epochs}", leave=False)

        for batch_index, (low, high, _, _) in enumerate(progress, start=1):
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(low)
                loss, terms = paired_rgb_restoration_loss(output, high)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for key, value in terms.items():
                train_sums[key] = train_sums.get(key, 0.0) + float(value.item())
            batches += 1

            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

        train_metrics = {k: v / max(1, batches) for k, v in train_sums.items()}

        # Multi-dataset validation
        macro_val, per_ds_val, all_val_rows = evaluate_all_val(model, val_loaders, evaluator, device)
        scheduler.step(macro_val["paired_loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = macro_val["paired_loss"] < best_loss
        if is_best:
            best_loss = macro_val["paired_loss"]
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, macro_val, args.base_channels, datasets_trained_list)
            save_comparisons_multi(model, val_loaders, device, compare_dir)

            # Export validation per-image metrics
            with open(per_image_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_val_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_val_rows)

            # Export validation per-dataset metrics
            with open(per_dataset_path, "w", newline="", encoding="utf-8") as f:
                fields = ["dataset", "paired_loss", "psnr", "ssim", "mse", "mae"]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for d_name, d_met in per_ds_val.items():
                    writer.writerow({"dataset": d_name, **{k: f"{d_met[k]:.4f}" for k in ("paired_loss", "psnr", "ssim", "mse", "mae")}})
                writer.writerow({"dataset": "OVERALL_MACRO", **{k: f"{macro_val[k]:.4f}" for k in ("paired_loss", "psnr", "ssim", "mse", "mae")}})
        else:
            bad_epochs += 1

        save_checkpoint(latest_path, model, optimizer, epoch, macro_val, args.base_channels, datasets_trained_list)

        per_ds_psnr_str = " | ".join(f"{d}: {per_ds_val[d]['psnr']:.2f}dB" for d in per_ds_val)
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_metrics.get('loss_total', 0.0):.4f} | "
            f"Val Loss: {macro_val['paired_loss']:.4f} | Val PSNR: {macro_val['psnr']:.2f} dB ({per_ds_psnr_str}) | "
            f"Val SSIM: {macro_val['ssim']:.4f} | LR: {current_lr:.2e} | Best: Ep {best_epoch} ({best_loss:.4f})"
        )

        row_history = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss_total", 0.0),
            "val_loss": macro_val["paired_loss"],
            "val_psnr": macro_val["psnr"],
            "val_ssim": macro_val["ssim"],
            "val_mae": macro_val["mae"],
            "lr": current_lr,
            "is_best": int(is_best),
        }
        for d in per_ds_val:
            row_history[f"{d}_val_psnr"] = per_ds_val[d]["psnr"]
            row_history[f"{d}_val_ssim"] = per_ds_val[d]["ssim"]
        history.append(row_history)

        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        if bad_epochs >= args.patience:
            print(f"[Early Stopping] Validation loss did not improve for {args.patience} epochs.")
            break

    print(f"\n[Training Complete] Best Checkpoint: {best_path} (Epoch {best_epoch}, Val Loss: {best_loss:.4f})")


if __name__ == "__main__":
    main()
