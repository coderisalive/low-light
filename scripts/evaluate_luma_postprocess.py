"""Validation-only sweep for luminance-domain post-processing.

Tests adaptive gamma and LAB-L CLAHE as a classical enhancement stage after the
pretrained SNR model. It does not retrain or modify the checkpoint. Candidate
selection uses a held-out split of our485 only; eval15 is not read by this script.
"""
import argparse
import csv
import itertools
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml
from tqdm import tqdm

from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from metrics.reference_metrics import ReferenceMetricsEvaluator
from models.snr_internal_reference_net import build_snr_internal_reference_net
from scripts.finetune_chrominance import paired_restoration_loss


def parse_float_list(value: str):
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of numbers.")
    return values


def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    _, _, height, width = x.shape
    pad_w = (multiple - width % multiple) % multiple
    pad_h = (multiple - height % multiple) % multiple
    if not (pad_w or pad_h):
        return x
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)


def load_model(checkpoint_path: Path, config_path: Path, device: torch.device):
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    with config_path.open("r", encoding="utf-8") as stream:
        fallback_config = yaml.safe_load(stream) or {}

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        config = checkpoint.get("config") or fallback_config
    else:
        state = checkpoint
        config = fallback_config

    model = build_snr_internal_reference_net(config).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def apply_luminance_postprocess(
    rgb_u8: np.ndarray,
    gamma: float,
    clahe_clip: float,
    saturation_gain: float,
    tile_grid: int,
) -> np.ndarray:
    """Apply gamma and optional CLAHE to CIE-Lab lightness, then optional HSV saturation."""
    if gamma <= 0:
        raise ValueError("Gamma must be positive.")
    if clahe_clip < 0:
        raise ValueError("CLAHE clip limit cannot be negative.")
    if saturation_gain < 0:
        raise ValueError("Saturation gain cannot be negative.")

    # Keep an exact identity candidate for a fair baseline.
    if abs(gamma - 1.0) < 1e-12 and clahe_clip == 0 and abs(saturation_gain - 1.0) < 1e-12:
        return rgb_u8.copy()

    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32) / 255.0
    lightness = np.clip(np.power(lightness, gamma), 0.0, 1.0)
    lightness_u8 = np.round(lightness * 255.0).astype(np.uint8)

    if clahe_clip > 0:
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip),
            tileGridSize=(int(tile_grid), int(tile_grid)),
        )
        lightness_u8 = clahe.apply(lightness_u8)

    lab[:, :, 0] = lightness_u8
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if abs(saturation_gain - 1.0) > 1e-12:
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) * float(saturation_gain)
        hsv[:, :, 1] = np.clip(np.round(sat), 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return result


def as_tensor(rgb_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(rgb_u8.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep gamma/CLAHE/saturation on a held-out our485 validation split."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True, help="LOL root containing our485/low and our485/high.")
    parser.add_argument("--config", default="configs/kaggle_gpu.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma_values", type=parse_float_list, default=[0.60, 0.70, 0.80, 0.90, 1.0])
    parser.add_argument("--clahe_values", type=parse_float_list, default=[0.0, 1.2, 1.5, 2.0])
    parser.add_argument("--saturation_values", type=parse_float_list, default=[1.0, 1.1, 1.2])
    parser.add_argument("--tile_grid", type=int, default=8)
    parser.add_argument("--max_val_images", type=int, default=None, help="Small smoke-test limit.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if not 0.05 <= args.validation_fraction < 0.5:
        raise ValueError("--validation_fraction must be in [0.05, 0.5).")
    if args.tile_grid < 1:
        raise ValueError("--tile_grid must be positive.")
    if args.max_val_images is not None and args.max_val_images < 1:
        raise ValueError("--max_val_images must be positive.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    print(f"[Device] {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    project = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project / config_path
    for required_path in (
        checkpoint_path,
        config_path,
        data_root / "our485" / "low",
        data_root / "our485" / "high",
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

    model = load_model(checkpoint_path, config_path, device)
    dataset = PairedEvaluationDataset(
        str(data_root / "our485" / "low"),
        str(data_root / "our485" / "high"),
        crop_size=None,
    )
    if len(dataset) < 2:
        raise RuntimeError(f"Need at least two paired images; found {len(dataset)}.")

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    n_val = max(1, min(len(indices) - 1, int(round(len(indices) * args.validation_fraction))))
    val_indices = sorted(indices[:n_val])
    if args.max_val_images is not None:
        val_indices = val_indices[:args.max_val_images]
    print(f"[Data] our485 pairs={len(dataset)}; validation images={len(val_indices)}; eval15 is not used.")

    cached = []
    with torch.inference_mode():
        for idx in tqdm(val_indices, desc="Model inference"):
            low, high, name = dataset[idx]
            input_batch = low.unsqueeze(0).to(device)
            _, _, height, width = input_batch.shape
            pred = model(pad_to_multiple(input_batch))[:, :, :height, :width].clamp(0.0, 1.0)
            pred_u8 = np.round(
                pred[0].permute(1, 2, 0).cpu().numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)
            cached.append({
                "name": str(name),
                "low": low.cpu(),
                "high": high.cpu(),
                "raw_u8": pred_u8,
                "target": high.unsqueeze(0).to(device),
            })

    candidates = {(1.0, 0.0, 1.0)}
    for gamma, clip, sat in itertools.product(
        args.gamma_values, args.clahe_values, args.saturation_values
    ):
        candidates.add((float(gamma), float(clip), float(sat)))
    candidates = sorted(candidates)

    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))
    scores = []
    best_candidate = None
    best_score = float("inf")
    best_outputs = None

    for gamma, clip, sat in tqdm(candidates, desc="Post-process validation sweep"):
        all_metrics = []
        candidate_outputs = []
        for item in cached:
            processed_u8 = apply_luminance_postprocess(
                item["raw_u8"], gamma, clip, sat, args.tile_grid
            )
            pred_t = as_tensor(processed_u8, device)
            target = item["target"].clamp(0.0, 1.0)
            metrics = evaluator.evaluate(pred_t, target)
            loss, terms = paired_restoration_loss(pred_t, target)
            all_metrics.append({
                **metrics,
                "paired_loss": float(loss.item()),
                "chroma_loss": float(terms["loss_chrominance"].item()),
                "contrast_loss": float(terms["loss_contrast"].item()),
            })
            candidate_outputs.append(processed_u8)

        summary = {
            "gamma": gamma,
            "clahe_clip": clip,
            "saturation_gain": sat,
            "psnr": float(np.mean([m["psnr"] for m in all_metrics])),
            "ssim": float(np.mean([m["ssim"] for m in all_metrics])),
            "mae": float(np.mean([m["mae"] for m in all_metrics])),
            "paired_loss": float(np.mean([m["paired_loss"] for m in all_metrics])),
            "chroma_loss": float(np.mean([m["chroma_loss"] for m in all_metrics])),
            "contrast_loss": float(np.mean([m["contrast_loss"] for m in all_metrics])),
        }
        scores.append(summary)
        if summary["paired_loss"] < best_score:
            best_score = summary["paired_loss"]
            best_candidate = summary
            best_outputs = candidate_outputs

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("/kaggle/working/experiments") / f"luma_postprocess_{stamp}"
        if Path("/kaggle/working").exists()
        else project / "experiments" / f"luma_postprocess_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    csv_path = output_dir / "candidate_metrics.csv"
    fields = [
        "gamma", "clahe_clip", "saturation_gain", "psnr", "ssim", "mae",
        "paired_loss", "chroma_loss", "contrast_loss",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(scores, key=lambda row: row["paired_loss"]))

    best_path = output_dir / "best_candidate.json"
    best_path.write_text(json.dumps(best_candidate, indent=2), encoding="utf-8")
    comparisons = output_dir / "best_comparisons"
    comparisons.mkdir()
    for item, output in zip(cached, best_outputs):
        h, w = output.shape[:2]
        pred_t = as_tensor(output, torch.device("cpu"))[0]
        low_t = item["low"]
        target_t = item["high"]
        raw_t = torch.from_numpy(
            item["raw_u8"].astype(np.float32).transpose(2, 0, 1) / 255.0
        )
        tiles = torch.stack([low_t, raw_t, pred_t, target_t])
        vutils.save_image(
            tiles,
            comparisons / f"{Path(item['name']).stem}.png",
            nrow=4,
            normalize=False,
        )

    identity = next(
        row for row in scores
        if row["gamma"] == 1.0 and row["clahe_clip"] == 0.0 and row["saturation_gain"] == 1.0
    )
    print("=" * 76)
    print(f"[Baseline model output] PSNR={identity['psnr']:.3f} | SSIM={identity['ssim']:.4f} | MAE={identity['mae']:.5f}")
    print(f"[Best validation candidate] gamma={best_candidate['gamma']} | CLAHE={best_candidate['clahe_clip']} | saturation={best_candidate['saturation_gain']}")
    print(f"[Best candidate] PSNR={best_candidate['psnr']:.3f} | SSIM={best_candidate['ssim']:.4f} | MAE={best_candidate['mae']:.5f}")
    print(f"[Best candidate] paired_loss={best_candidate['paired_loss']:.5f} | chroma={best_candidate['chroma_loss']:.5f} | contrast={best_candidate['contrast_loss']:.5f}")
    print(f"[CSV] {csv_path}")
    print(f"[Best config] {best_path}")
    print(f"[Comparisons] {comparisons}")
    print("[Protocol] Candidate selection used only the held-out our485 validation split; eval15 was not loaded.")


if __name__ == "__main__":
    main()
