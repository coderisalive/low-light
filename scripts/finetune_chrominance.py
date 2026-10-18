"""
Paired chrominance-restoration fine-tuning for SNRInternalReferenceNet.

Only the existing enhancement_head parameters are optimized by default. This keeps
the pretrained SNR/reliability/reference features fixed, reduces catastrophic
forgetting, and focuses the paired objective on RGB, luminance, chrominance and
luminance-gradient fidelity.

The paired train/validation split is built from our485 only. eval15 is reserved
for a separate final evaluation through scripts/evaluate.py.
"""
import argparse
import csv
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils
from tqdm import tqdm

from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from metrics.reference_metrics import ReferenceMetricsEvaluator
from models.enhancement_head import ColorSpaceConverter
from models.snr_internal_reference_net import build_snr_internal_reference_net


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def paired_restoration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    w_rgb: float = 1.0,
    w_luminance: float = 0.5,
    w_chrominance: float = 2.0,
    w_gradient: float = 0.2,
    w_color_angle: float = 0.1,
    w_contrast: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Paired RGB/chroma loss with local luminance-contrast preservation."""
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    pred_y, pred_cbcr = ColorSpaceConverter.rgb_to_ycbcr(prediction)
    target_y, target_cbcr = ColorSpaceConverter.rgb_to_ycbcr(target)

    l_rgb = F.l1_loss(prediction, target)
    l_luminance = F.l1_loss(pred_y, target_y)
    l_chrominance = F.l1_loss(pred_cbcr, target_cbcr)

    # Match local luminance contrast at patch scale to help prevent flat results.
    contrast_patch = min(16, prediction.shape[-2], prediction.shape[-1])

    def local_luminance_std(y: torch.Tensor) -> torch.Tensor:
        local_mean = F.avg_pool2d(y, kernel_size=contrast_patch, stride=contrast_patch)
        local_mean_sq = F.avg_pool2d(y.square(), kernel_size=contrast_patch, stride=contrast_patch)
        return torch.sqrt((local_mean_sq - local_mean.square()).clamp_min(0.0) + 1e-6)

    l_contrast = F.l1_loss(local_luminance_std(pred_y), local_luminance_std(target_y))

    pred_dx = pred_y[:, :, :, 1:] - pred_y[:, :, :, :-1]
    targ_dx = target_y[:, :, :, 1:] - target_y[:, :, :, :-1]
    pred_dy = pred_y[:, :, 1:, :] - pred_y[:, :, :-1, :]
    targ_dy = target_y[:, :, 1:, :] - target_y[:, :, :-1, :]
    l_gradient = F.l1_loss(pred_dx, targ_dx) + F.l1_loss(pred_dy, targ_dy)

    pred_unit = F.normalize(prediction, p=2, dim=1, eps=1e-6)
    target_unit = F.normalize(target, p=2, dim=1, eps=1e-6)
    cosine = (pred_unit * target_unit).sum(dim=1).clamp(-1.0, 1.0)
    l_color_angle = (1.0 - cosine).mean()

    total = (
        w_rgb * l_rgb
        + w_luminance * l_luminance
        + w_chrominance * l_chrominance
        + w_gradient * l_gradient
        + w_color_angle * l_color_angle
        + w_contrast * l_contrast
    )
    terms = {
        "loss_total": total,
        "loss_rgb": l_rgb,
        "loss_luminance": l_luminance,
        "loss_chrominance": l_chrominance,
        "loss_gradient": l_gradient,
        "loss_color_angle": l_color_angle,
        "loss_contrast": l_contrast,
    }
    return total, terms


def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    _, _, height, width = x.shape
    pad_w = (multiple - width % multiple) % multiple
    pad_h = (multiple - height % multiple) % multiple
    if not (pad_w or pad_h):
        return x
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    evaluator: ReferenceMetricsEvaluator,
    device: torch.device,
    comparison_dir: Path = None,
    max_comparisons: int = 5,
    loss_weights: Dict[str, float] = None,
) -> Tuple[Dict[str, float], list]:
    model.eval()
    rows = []
    if comparison_dir is not None:
        comparison_dir.mkdir(parents=True, exist_ok=True)

    for index, (low, high, name) in enumerate(tqdm(loader, desc="Paired validation", leave=False)):
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)
        _, _, h, w = low.shape
        prediction = model(pad_to_multiple(low))[:, :, :h, :w].clamp(0.0, 1.0)
        target = high.clamp(0.0, 1.0)
        metrics = evaluator.evaluate(prediction, target)
        paired_loss, loss_terms = paired_restoration_loss(
            prediction, target, **(loss_weights or {})
        )
        image_name = name[0] if isinstance(name, (list, tuple)) else str(name)
        rows.append({
            "image": image_name,
            **metrics,
            "paired_loss": float(paired_loss.item()),
            "loss_chrominance": float(loss_terms["loss_chrominance"].item()),
            "loss_contrast": float(loss_terms["loss_contrast"].item()),
        })

        if comparison_dir is not None and index < max_comparisons:
            residual = ((prediction - high).abs() * 4.0).clamp(0.0, 1.0)
            tiles = torch.cat(
                [low.cpu(), prediction.cpu(), high.cpu(), residual.cpu()], dim=0
            )
            vutils.save_image(
                tiles,
                comparison_dir / f"{Path(image_name).stem}_comparison.png",
                nrow=4,
                normalize=False,
            )

    if not rows:
        raise RuntimeError("Validation loader produced zero paired images.")
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("psnr", "ssim", "mse", "mae", "paired_loss",
                    "loss_chrominance", "loss_contrast")
    }
    return summary, rows


def load_checkpoint(path: Path, fallback_config: dict, device: torch.device):
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        config = checkpoint.get("config") or fallback_config
    else:
        state = checkpoint
        config = fallback_config

    model = build_snr_internal_reference_net(config).to(device)
    model.load_state_dict(state, strict=True)
    return model, config, checkpoint


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    epoch: int,
    metrics: dict,
    source_checkpoint: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_type": "snr_internal_reference",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "fine_tune_mode": "enhancement_head_only",
            "fine_tuned_from": source_checkpoint,
            "validation_metrics": metrics,
        },
        path,
    )


def write_per_image_metrics(path: Path, epoch: int, rows: list) -> None:
    """Save metrics for the exact checkpoint currently selected as best."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[
                "epoch", "image", "psnr", "ssim", "mse", "mae",
                "paired_loss", "loss_chrominance", "loss_contrast",
            ]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "epoch": epoch,
                "image": row["image"],
                "psnr": row["psnr"],
                "ssim": row["ssim"],
                "mse": row["mse"],
                "mae": row["mae"],
                "paired_loss": row["paired_loss"],
                "loss_chrominance": row["loss_chrominance"],
                "loss_contrast": row["loss_contrast"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired RGB/chrominance fine-tuning for SNRInternalReferenceNet"
    )
    parser.add_argument("--config", default="configs/kaggle_gpu.yaml")
    parser.add_argument("--data_root", required=True,
                        help="LOL root containing our485/low, our485/high and eval15/")
    parser.add_argument("--checkpoint", required=True,
                        help="Existing trained SNR checkpoint; it is never overwritten")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="Optional smoke-test limit; omit for full training")
    parser.add_argument("--w_rgb", type=float, default=1.0)
    parser.add_argument("--w_luminance", type=float, default=0.5)
    parser.add_argument("--w_chrominance", type=float, default=2.0)
    parser.add_argument("--w_gradient", type=float, default=0.2)
    parser.add_argument("--w_color_angle", type=float, default=0.1)
    parser.add_argument("--w_contrast", type=float, default=0.2)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.crop_size < 16:
        raise ValueError("epochs/batch_size must be positive and crop_size >= 16.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("--max_train_batches must be positive when supplied.")
    if not 0.05 <= args.validation_fraction < 0.5:
        raise ValueError("--validation_fraction must be in [0.05, 0.5).")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    project = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project / config_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    low_dir = data_root / "our485" / "low"
    high_dir = data_root / "our485" / "high"
    required = [
        low_dir, high_dir,
        data_root / "eval15" / "low", data_root / "eval15" / "high",
    ]
    if not all(path.is_dir() for path in required):
        raise FileNotFoundError(
            f"Expected LOL directories under {data_root}: "
            "our485/low, our485/high, eval15/low, eval15/high"
        )

    with config_path.open("r", encoding="utf-8") as stream:
        fallback_config = yaml.safe_load(stream) or {}

    model, config, original_ckpt = load_checkpoint(checkpoint_path, fallback_config, device)
    print(f"[Model] {model.count_parameters():,} trainable parameters")
    print(f"[Init] Loaded pretrained weights from {checkpoint_path}")

    # Freeze all feature/reliability/reference layers; only fine-tune the existing
    # enhancement head, so model shapes and checkpoint key structure stay unchanged.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("enhancement_head.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No enhancement_head parameters were selected for fine-tuning.")
    print(f"[Fine-tune] Trainable parameters: {sum(p.numel() for p in trainable):,} (enhancement head only)")

    paired_train_all = PairedEvaluationDataset(str(low_dir), str(high_dir), crop_size=args.crop_size)
    paired_val_all = PairedEvaluationDataset(str(low_dir), str(high_dir), crop_size=None)
    if len(paired_train_all) < 2 or len(paired_val_all) != len(paired_train_all):
        raise RuntimeError(
            f"Expected matched low/high pairs in our485; got train={len(paired_train_all)}, "
            f"validation_source={len(paired_val_all)}."
        )

    indices = list(range(len(paired_train_all)))
    random.Random(args.seed).shuffle(indices)
    n_val = max(1, min(len(indices) - 1, int(round(len(indices) * args.validation_fraction))))
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])

    train_ds = Subset(paired_train_all, train_indices)
    val_ds = Subset(paired_val_all, val_indices)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Empty paired train/validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=False, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    print(
        f"[Data] our485 pairs={len(paired_train_all)}; train={len(train_ds)}; "
        f"held-out validation={len(val_ds)}; eval15 reserved for final test."
    )

    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))
    loss_weights = {
        "w_rgb": args.w_rgb,
        "w_luminance": args.w_luminance,
        "w_chrominance": args.w_chrominance,
        "w_gradient": args.w_gradient,
        "w_color_angle": args.w_color_angle,
        "w_contrast": args.w_contrast,
    }
    output_root = Path("/kaggle/working/experiments") if Path("/kaggle/working").exists() else project / "experiments"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_root / f"chroma_finetune_{stamp}"
    comparisons_dir = exp_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=False)

    history_csv = exp_dir / "training_history.csv"
    per_image_csv = exp_dir / "validation_per_image.csv"
    best_checkpoint = exp_dir / "best_checkpoint.pth"
    latest_checkpoint = exp_dir / "latest_checkpoint.pth"
    exported_best = output_root.parent / f"{exp_dir.name}_best_checkpoint.pth"

    # Record initial checkpoint performance on the same held-out validation split.
    model.eval()
    initial_metrics, initial_rows = evaluate(
        model, val_loader, evaluator, device,
        comparison_dir=comparisons_dir, max_comparisons=8,
        loss_weights=loss_weights
    )
    print(
        f"[Initial held-out] PSNR={initial_metrics['psnr']:.3f} dB | "
        f"SSIM={initial_metrics['ssim']:.4f} | MAE={initial_metrics['mae']:.5f}"
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
        threshold=1e-5, threshold_mode="abs", min_lr=1e-7,
    )
    save_checkpoint(
        exp_dir / "initial_checkpoint.pth", model, optimizer, scheduler,
        config, 0, initial_metrics, str(checkpoint_path)
    )
    write_per_image_metrics(per_image_csv, 0, initial_rows)
    best_paired_loss = initial_metrics["paired_loss"]
    best_metrics = dict(initial_metrics)
    best_epoch = 0
    bad_epochs = 0
    # Initial weights are the fallback best. Save to a new experiment folder only.
    save_checkpoint(
        best_checkpoint, model, optimizer, scheduler, config, 0,
        best_metrics, str(checkpoint_path)
    )
    shutil.copy2(best_checkpoint, exported_best)

    history = [{
        "epoch": 0,
        "phase": "initial",
        "train_loss": "",
        "loss_rgb": "",
        "loss_luminance": "",
        "loss_chrominance": "",
        "loss_gradient": "",
        "loss_color_angle": "",
        "loss_contrast": "",
        "val_psnr": initial_metrics["psnr"],
        "val_ssim": initial_metrics["ssim"],
        "val_mse": initial_metrics["mse"],
        "val_mae": initial_metrics["mae"],
        "val_paired_loss": initial_metrics["paired_loss"],
        "val_chrominance_loss": initial_metrics["loss_chrominance"],
        "val_contrast_loss": initial_metrics["loss_contrast"],
        "lr": args.lr,
        "best_paired_loss": best_paired_loss,
    }]

    for epoch in range(1, args.epochs + 1):
        # Frozen feature extractor stays in eval mode; only head modules train.
        model.eval()
        model.enhancement_head.train()
        sums: Dict[str, float] = {}
        steps = 0
        pbar = tqdm(train_loader, desc=f"Fine-tune {epoch:02d}/{args.epochs}", leave=False)

        for batch_index, (low, high, _) in enumerate(pbar, start=1):
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)

            # Synchronized, label-preserving geometric augmentation.
            if random.random() < 0.5:
                low, high = torch.flip(low, dims=[3]), torch.flip(high, dims=[3])
            if random.random() < 0.5:
                low, high = torch.flip(low, dims=[2]), torch.flip(high, dims=[2])

            optimizer.zero_grad(set_to_none=True)
            prediction = model(pad_to_multiple(low))
            prediction = prediction[:, :, :low.shape[-2], :low.shape[-1]]
            loss, terms = paired_restoration_loss(
                prediction, high,
                w_rgb=args.w_rgb,
                w_luminance=args.w_luminance,
                w_chrominance=args.w_chrominance,
                w_gradient=args.w_gradient,
                w_color_angle=args.w_color_angle,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite paired loss at epoch {epoch}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            for parameter in trainable:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Non-finite gradient at epoch {epoch}.")
            optimizer.step()

            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().item())
            steps += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                rgb=f"{terms['loss_rgb'].item():.4f}",
                chroma=f"{terms['loss_chrominance'].item():.4f}",
            )
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                print(f"[Smoke test] Stopping after {batch_index} training batches as requested.")
                break

        if steps == 0:
            raise RuntimeError("Training DataLoader produced no batches.")
        means = {key: value / steps for key, value in sums.items()}

        model.eval()
        val_metrics, val_rows = evaluate(
            model, val_loader, evaluator, device, loss_weights=loss_weights
        )
        scheduler.step(val_metrics["paired_loss"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        improved = val_metrics["paired_loss"] < best_paired_loss - 1e-5
        if improved:
            best_paired_loss = val_metrics["paired_loss"]
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                best_checkpoint, model, optimizer, scheduler, config, epoch,
                best_metrics, str(checkpoint_path)
            )
            shutil.copy2(best_checkpoint, exported_best)
            # Keep per-image metrics and visualizations aligned with the best checkpoint.
            write_per_image_metrics(per_image_csv, epoch, val_rows)
            for old in comparisons_dir.glob("*.png"):
                old.unlink()
            _, _ = evaluate(
                model, val_loader, evaluator, device,
                comparison_dir=comparisons_dir, max_comparisons=8,
                loss_weights=loss_weights
            )
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "phase": "fine_tune",
            "train_loss": means["loss_total"],
            "loss_rgb": means["loss_rgb"],
            "loss_luminance": means["loss_luminance"],
            "loss_chrominance": means["loss_chrominance"],
            "loss_gradient": means["loss_gradient"],
            "loss_color_angle": means["loss_color_angle"],
            "loss_contrast": means["loss_contrast"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_paired_loss": val_metrics["paired_loss"],
            "val_chrominance_loss": val_metrics["loss_chrominance"],
            "val_contrast_loss": val_metrics["loss_contrast"],
            "lr": lr_now,
            "best_paired_loss": best_paired_loss,
        }
        history.append(row)
        save_checkpoint(
            latest_checkpoint, model, optimizer, scheduler, config, epoch,
            val_metrics, str(checkpoint_path)
        )
        with history_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={means['loss_total']:.5f} | "
            f"RGB={means['loss_rgb']:.5f} | Chroma={means['loss_chrominance']:.5f} | "
            f"Val objective={val_metrics['paired_loss']:.5f} | "
            f"val chroma={val_metrics['loss_chrominance']:.5f} | "
            f"val contrast={val_metrics['loss_contrast']:.5f} | "
            f"PSNR={val_metrics['psnr']:.3f} dB | SSIM={val_metrics['ssim']:.4f} | "
            f"LR={lr_now:.2e} | best={best_epoch} | patience={bad_epochs}/{args.patience}"
            + (" [BEST]" if improved else "")
        )

        if not improved and bad_epochs >= args.patience:
            print(f"[Early stopping] Paired RGB/chroma/contrast objective did not improve for {bad_epochs} epochs.")
            break

    print("=" * 78)
    print(f"[Complete] Best epoch: {best_epoch}")
    print(f"[Complete] Best held-out paired objective: {best_paired_loss:.5f}")
    print(f"[Complete] PSNR at selected checkpoint: {best_metrics['psnr']:.3f} dB")
    print(f"[Complete] SSIM at selected checkpoint: {best_metrics['ssim']:.4f}")
    print(f"[Complete] Chroma L1 at selected checkpoint: {best_metrics['loss_chrominance']:.5f}")
    print(f"[Complete] Local contrast loss at selected checkpoint: {best_metrics['loss_contrast']:.5f}")
    print(f"[Complete] MAE at selected checkpoint: {best_metrics['mae']:.5f}")
    print(f"[Complete] Best checkpoint: {best_checkpoint}")
    print(f"[Complete] Portable best checkpoint: {exported_best}")
    print(f"[Complete] History: {history_csv}")
    print(f"[Complete] Per-image validation: {per_image_csv}")
    print(f"[Complete] Comparisons: {comparisons_dir}")
    print("[Protocol] eval15 was not used during training or checkpoint selection.")
    print("=" * 78)


if __name__ == "__main__":
    main()
