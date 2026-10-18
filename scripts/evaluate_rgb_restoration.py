"""Evaluate PairedRGBRestorationNet on paired low/high images."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from datasets.paired_evaluation_dataset import PairedEvaluationDataset
from metrics.reference_metrics import ReferenceMetricsEvaluator
from models.paired_rgb_restoration_net import PairedRGBRestorationNet


def to_rgb8(x: torch.Tensor) -> np.ndarray:
    return (x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def save_montage(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    images = [Image.fromarray(array.astype(np.uint8), mode="RGB") for _, array in panels]
    if not images or any(image.size != images[0].size for image in images):
        raise ValueError("Montage images must have identical dimensions.")
    w, h = images[0].size
    label_h = 34
    canvas = Image.new("RGB", (w * len(images), h + label_h), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, ((label, _), img) in enumerate(zip(panels, images)):
        x = i * w
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        draw.text((x + max(3, (w - tw) // 2), (label_h - th) // 2 - box[1]),
                  label, fill="white", font=font)
        canvas.paste(img, (x, label_h))
    canvas.save(path)


def load_model(checkpoint_path: Path, device: torch.device):
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    model = PairedRGBRestorationNet(base_channels=int(cfg.get("base_channels", 24))).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the paired-supervised RGB U-Net.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--low_dir", required=True)
    parser.add_argument("--high_dir", required=True)
    parser.add_argument("--output_csv", default="experiments/paired_rgb_eval.csv")
    parser.add_argument("--comparison_dir", default="experiments/paired_rgb_comparisons")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"[Device] {device}" + (f" — {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    model = load_model(checkpoint, device)
    print(f"[Model] PairedRGBRestorationNet | {model.count_parameters():,} parameters")
    dataset = PairedEvaluationDataset(args.low_dir, args.high_dir, crop_size=None)
    if len(dataset) == 0:
        raise RuntimeError("No matching paired images were found.")
    limit = min(len(dataset), args.max_images) if args.max_images is not None else len(dataset)
    if limit < len(dataset):
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(limit)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    evaluator = ReferenceMetricsEvaluator(use_lpips=False, device=str(device))

    comparison_dir = Path(args.comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for low, high, names in tqdm(loader, desc="Evaluating paired RGB U-Net"):
        low = low.to(device)
        high = high.to(device)
        output = model(low).clamp(0, 1)
        metrics = evaluator.evaluate(output, high.clamp(0, 1))
        name = names[0] if isinstance(names, (tuple, list)) else str(names)
        rows.append({"image": name, **metrics})
        residual = ((output[0] - high[0]).abs() * 4.0).clamp(0, 1)
        save_montage(
            comparison_dir / f"{Path(name).stem}.png",
            [
                ("Low-light Input", to_rgb8(low[0])),
                ("RGB U-Net Output", to_rgb8(output[0])),
                ("Ground Truth", to_rgb8(high[0])),
                ("Residual x4", to_rgb8(residual)),
            ],
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "psnr", "ssim", "mse", "mae"]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ("psnr", "ssim", "mse", "mae")}
    print(f"[Metrics] {len(rows)} images | PSNR={summary['psnr']:.3f} dB | SSIM={summary['ssim']:.4f} | MAE={summary['mae']:.5f}")
    print(f"[CSV] {output_csv}")
    print(f"[Labeled comparisons] {comparison_dir}")


if __name__ == "__main__":
    main()
