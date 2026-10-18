"""Per-image adaptive luminance enhancement sweep.

Unlike a single global gamma/CLAHE setting, this estimates a conservative gamma
from each model output's mean lightness, optionally blends mild LAB-L CLAHE, and
optionally adjusts saturation. It evaluates on held-out our485 only and selects
only candidates whose PSNR is no worse than baseline and SSIM drop is <= 0.005.
If no candidate satisfies these constraints, identity is selected.
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
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from metrics.reference_metrics import ReferenceMetricsEvaluator
from models.snr_internal_reference_net import build_snr_internal_reference_net
from scripts.finetune_chrominance import paired_restoration_loss


def parse_float_list(value: str) -> list[float]:
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


def tensor_to_rgb8(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0
        + 0.5
    ).clip(0, 255).astype(np.uint8)




def save_labeled_comparison(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    label_height: int = 34,
) -> None:
    """Save an exact-value, side-by-side montage with a title bar per panel."""
    if not panels:
        raise ValueError("At least one comparison panel is required.")

    images = [Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB") for _, array in panels]
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("Comparison panels must have identical dimensions.")

    canvas = Image.new(
        "RGB",
        (width * len(images), height + label_height),
        color=(28, 28, 28),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for index, ((label, _), image) in enumerate(zip(panels, images)):
        x = index * width
        draw.rectangle((x, 0, x + width - 1, label_height - 1), fill=(35, 35, 35))
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        tx = x + max(4, (width - text_width) // 2)
        ty = max(0, (label_height - text_height) // 2 - text_box[1])
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
        canvas.paste(image, (x, label_height))
        if index:
            draw.line((x, 0, x, label_height + height), fill=(100, 100, 100), width=1)

    canvas.save(path)


def rgb8_to_tensor(rgb8: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(rgb8.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def adaptive_luma_enhance(
    rgb8: np.ndarray,
    target_mean: float,
    gamma_strength: float,
    clahe_clip: float,
    clahe_strength: float,
    saturation_gain: float,
    tile_grid: int,
) -> np.ndarray:
    """Estimate gamma per image from LAB lightness, then blend optional CLAHE."""
    if not 0.0 <= target_mean < 1.0:
        raise ValueError("target_mean must be in [0, 1).")
    if not 0.0 <= gamma_strength <= 1.0:
        raise ValueError("gamma_strength must be in [0, 1].")
    if clahe_clip < 0 or not 0.0 <= clahe_strength <= 1.0:
        raise ValueError("CLAHE clip/strength is invalid.")
    if saturation_gain < 0 or tile_grid < 1:
        raise ValueError("Saturation gain must be non-negative and tile grid positive.")

    # Identity is kept as the comparison baseline and fallback candidate.
    if gamma_strength == 0.0 and clahe_strength == 0.0 and abs(saturation_gain - 1.0) < 1e-8:
        return rgb8.copy()

    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32) / 255.0
    mean_lightness = float(np.mean(lightness))

    # Estimate gamma individually. Brighten dark outputs, but never darken an
    # already bright output; gamma is bounded to avoid aggressive lifting.
    if mean_lightness < 1e-5 or target_mean <= 1e-5:
        gamma = 1.0
    else:
        gamma = float(np.clip(np.log(target_mean) / np.log(max(mean_lightness, 1e-5)), 0.55, 1.0))

    gamma_l = np.power(np.clip(lightness, 0.0, 1.0), gamma)
    mixed_l = (1.0 - gamma_strength) * lightness + gamma_strength * gamma_l
    l_u8 = np.round(np.clip(mixed_l, 0.0, 1.0) * 255.0).astype(np.uint8)

    if clahe_clip > 0.0 and clahe_strength > 0.0:
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip),
            tileGridSize=(int(tile_grid), int(tile_grid)),
        )
        clahe_l = clahe.apply(l_u8)
        l_u8 = np.round(
            (1.0 - clahe_strength) * l_u8.astype(np.float32)
            + clahe_strength * clahe_l.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

    lab[:, :, 0] = l_u8
    output = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if abs(saturation_gain - 1.0) > 1e-8:
        hsv = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) * saturation_gain
        hsv[:, :, 1] = np.round(saturation).clip(0, 255).astype(np.uint8)
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return output


def summarize_candidate(outputs, cached, evaluator, device):
    rows = []
    for output, item in zip(outputs, cached):
        pred = rgb8_to_tensor(output, device)
        target = item["target"].clamp(0.0, 1.0)
        metrics = evaluator.evaluate(pred, target)
        loss, terms = paired_restoration_loss(pred, target)
        rows.append({
            "image": item["name"],
            **metrics,
            "paired_loss": float(loss.item()),
            "chroma_loss": float(terms["loss_chrominance"].item()),
            "contrast_loss": float(terms["loss_contrast"].item()),
        })
    return {
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
        "paired_loss": float(np.mean([row["paired_loss"] for row in rows])),
        "chroma_loss": float(np.mean([row["chroma_loss"] for row in rows])),
        "contrast_loss": float(np.mean([row["contrast_loss"] for row in rows])),
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-image adaptive gamma/CLAHE validation sweep.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True, help="LOL root containing our485/low and our485/high.")
    parser.add_argument("--config", default="configs/kaggle_gpu.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_means", type=parse_float_list, default=[0.34, 0.38, 0.42])
    parser.add_argument("--gamma_strengths", type=parse_float_list, default=[0.35, 0.65, 1.0])
    parser.add_argument("--clahe_values", type=parse_float_list, default=[0.0, 1.0, 1.5])
    parser.add_argument("--clahe_strengths", type=parse_float_list, default=[0.0, 0.3])
    parser.add_argument("--saturation_values", type=parse_float_list, default=[1.0, 1.08])
    parser.add_argument("--ssim_tolerance", type=float, default=0.005)
    parser.add_argument("--tile_grid", type=int, default=8)
    parser.add_argument("--max_val_images", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if not 0.05 <= args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0.05, 0.5).")
    if args.ssim_tolerance < 0 or args.tile_grid < 1:
        raise ValueError("SSIM tolerance must be non-negative and tile grid positive.")
    if args.max_val_images is not None and args.max_val_images < 1:
        raise ValueError("--max_val_images must be positive when supplied.")

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
        checkpoint_path, config_path,
        data_root / "our485" / "low", data_root / "our485" / "high",
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
            x = low.unsqueeze(0).to(device)
            _, _, h, w = x.shape
            pred = model(pad_to_multiple(x))[:, :, :h, :w].clamp(0.0, 1.0)
            cached.append({
                "name": str(name),
                "low": low.cpu(),
                "raw_u8": tensor_to_rgb8(pred[0]),
                "target": high.unsqueeze(0).to(device),
            })

    # Candidate identity is measured exactly once and is always eligible as fallback.
    identity_outputs = [item["raw_u8"] for item in cached]
    identity_metrics, _ = summarize_candidate(identity_outputs, cached, ReferenceMetricsEvaluator(device=str(device)), device)
    candidates = []
    for target_mean, gamma_strength, clahe_clip, clahe_strength, saturation in itertools.product(
        args.target_means, args.gamma_strengths, args.clahe_values,
        args.clahe_strengths, args.saturation_values,
    ):
        if clahe_clip == 0.0 and clahe_strength > 0.0:
            continue
        if (
            gamma_strength == 0.0 and clahe_strength == 0.0
            and abs(saturation - 1.0) < 1e-8
        ):
            continue
        candidates.append({
            "target_mean": float(target_mean),
            "gamma_strength": float(gamma_strength),
            "clahe_clip": float(clahe_clip),
            "clahe_strength": float(clahe_strength),
            "saturation_gain": float(saturation),
        })

    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))
    scores = [{
        "candidate": "identity",
        "target_mean": "",
        "gamma_strength": 0.0,
        "clahe_clip": 0.0,
        "clahe_strength": 0.0,
        "saturation_gain": 1.0,
        **identity_metrics,
        "eligible": True,
    }]
    best_outputs = identity_outputs
    best_config = {
        "candidate": "identity",
        "target_mean": None,
        "gamma_strength": 0.0,
        "clahe_clip": 0.0,
        "clahe_strength": 0.0,
        "saturation_gain": 1.0,
        **identity_metrics,
    }
    eligible_losses = [identity_metrics["paired_loss"]]

    for config in tqdm(candidates, desc="Adaptive post-process sweep"):
        outputs = [
            adaptive_luma_enhance(
                item["raw_u8"],
                config["target_mean"],
                config["gamma_strength"],
                config["clahe_clip"],
                config["clahe_strength"],
                config["saturation_gain"],
                args.tile_grid,
            )
            for item in cached
        ]
        metrics, _ = summarize_candidate(outputs, cached, evaluator, device)
        eligible = (
            metrics["psnr"] >= identity_metrics["psnr"]
            and metrics["ssim"] >= identity_metrics["ssim"] - args.ssim_tolerance
        )
        scores.append({
            **config,
            **metrics,
            "candidate": "adaptive",
            "eligible": eligible,
        })
        if eligible and metrics["paired_loss"] < min(eligible_losses):
            eligible_losses.append(metrics["paired_loss"])
            best_outputs = outputs
            best_config = {**config, **metrics, "candidate": "adaptive"}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif Path("/kaggle/working").exists():
        output_dir = Path("/kaggle/working/experiments") / f"adaptive_luma_{stamp}"
    else:
        output_dir = project / "experiments" / f"adaptive_luma_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    fields = [
        "candidate", "target_mean", "gamma_strength", "clahe_clip",
        "clahe_strength", "saturation_gain", "psnr", "ssim", "mae",
        "paired_loss", "chroma_loss", "contrast_loss", "eligible",
    ]
    csv_path = output_dir / "candidate_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(scores, key=lambda row: (not bool(row["eligible"]), row["paired_loss"])))

    config_path_out = output_dir / "selected_config.json"
    config_path_out.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    comparisons = output_dir / "selected_comparisons"
    comparisons.mkdir()
    for item, output in zip(cached, best_outputs):
        save_labeled_comparison(
            comparisons / f"{Path(item['name']).stem}.png",
            [
                ("Low-light Input", tensor_to_rgb8(item["low"])),
                ("Original Model Output", item["raw_u8"]),
                ("Adaptive Output", output),
                ("Ground Truth", tensor_to_rgb8(item["target"][0].cpu())),
            ],
        )

    print("=" * 76)
    print(f"[Baseline] PSNR={identity_metrics['psnr']:.3f} | SSIM={identity_metrics['ssim']:.4f} | MAE={identity_metrics['mae']:.5f}")
    print(f"[Selected] {best_config['candidate']} | PSNR={best_config['psnr']:.3f} | SSIM={best_config['ssim']:.4f} | MAE={best_config['mae']:.5f}")
    print(f"[Selected] paired_loss={best_config['paired_loss']:.5f} | chroma={best_config['chroma_loss']:.5f} | contrast={best_config['contrast_loss']:.5f}")
    print(f"[Selection] Candidate must have PSNR >= baseline and SSIM >= baseline - {args.ssim_tolerance:.4f}.")
    print(f"[CSV] {csv_path}")
    print(f"[Selected config] {config_path_out}")
    print(f"[Comparisons] {comparisons}")
    print("[Protocol] Only held-out our485 was loaded; eval15 was not read.")
    if best_config["candidate"] == "identity":
        print("[Decision] No adaptive candidate met the quality constraints; identity fallback selected.")


if __name__ == "__main__":
    main()
