"""Run PairedRGBRestorationNet on an image or directory of images."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from scripts.evaluate_rgb_restoration import load_model


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance low-light images with the paired RGB U-Net.")
    parser.add_argument("--input", required=True, help="Input image path or directory.")
    parser.add_argument("--checkpoint", required=True, help="PairedRGBRestorationNet checkpoint.")
    parser.add_argument("--output", default="outputs/enhanced", help="Directory for enhanced images.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path] if input_path.suffix.lower() in VALID_EXTENSIONS else []
    elif input_path.is_dir():
        image_paths = sorted(p for p in input_path.iterdir()
                             if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not image_paths:
        raise RuntimeError(f"No supported images found at: {input_path}")

    model = load_model(Path(args.checkpoint), device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Device] {device}")
    print(f"[Model] PairedRGBRestorationNet | {model.count_parameters():,} parameters")
    print(f"[Input] {len(image_paths)} image(s)")

    with torch.inference_mode():
        for path in image_paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            x = TF.to_tensor(rgb).unsqueeze(0).to(device)
            enhanced = model(x).clamp(0, 1)[0].cpu()
            output = TF.to_pil_image(enhanced)
            output_path = out_dir / f"{path.stem}_enhanced.png"
            output.save(output_path)

            comparison = Image.new("RGB", (rgb.width * 2, rgb.height))
            comparison.paste(rgb, (0, 0))
            comparison.paste(output.resize(rgb.size), (rgb.width, 0))
            comparison.save(out_dir / f"{path.stem}_comparison.png")
            print(f"[Saved] {output_path}")

    print(f"[Complete] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
