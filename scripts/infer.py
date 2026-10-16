"""
Inference & Visualization Script for LumaGuard-Net.
Supports:
- Single image or directory of images
- Automatic model architecture reconstruction from checkpoint config
- Automatic tiling / downscaling for ultra-high-resolution images (e.g. 4000x4000)
  to run smoothly without CPU Out-Of-Memory (OOM)
- Diagnostic outputs:
  - Low-light original
  - Enhanced RGB
  - Illumination map L
  - Noise uncertainty map N
  - Reliability map Q
  - Absolute difference
  - Side-by-side comparison
"""

import argparse
from pathlib import Path
import os
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from models.lumaguard_net import LumaGuardNet, build_lumaguard_net
from models.snr_internal_reference_net import (
    SNRInternalReferenceNet,
    build_snr_internal_reference_net,
)


def process_image(
    model: nn.Module,
    img_path: Path,
    out_dir: Path,
    device: torch.device,
    save_diagnostics: bool = True,
    max_dimension: int = 1200,
) -> None:
    with Image.open(img_path) as pil_img:
        pil_img = pil_img.convert("RGB")
        orig_w, orig_h = pil_img.size

        # For ultra-high resolution images on CPU, scale down while preserving aspect ratio
        if max(orig_w, orig_h) > max_dimension:
            scale = max_dimension / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            print(f"[Notice] Resized {img_path.name} from ({orig_w}x{orig_h}) to ({new_w}x{new_h}) for safe CPU inference.")

        w, h = pil_img.size

    # Ensure dimensions are divisible by 16 for patch grid matching
    pad_w = (16 - (w % 16)) % 16
    pad_h = (16 - (h % 16)) % 16

    tensor_in = TF.to_tensor(pil_img).unsqueeze(0).to(device)
    if pad_w > 0 or pad_h > 0:
        tensor_in = torch.nn.functional.pad(tensor_in, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        diag = model.forward_diagnostic(tensor_in)

        # Unpad
        unpad_keys = [
            "enhanced",
            "illumination",
            "noise_uncertainty",
            "noise_proxy",
            "reliability_map",
            "match_confidence",
            "ref_reliability",
            "transfer_gate",
        ]
        for key in unpad_keys:
            if key in diag and diag[key] is not None:
                diag[key] = diag[key][:, :, :h, :w]

    orig = TF.to_tensor(pil_img).unsqueeze(0).to(device)
    enhanced = diag["enhanced"]

    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = img_path.stem

    # Save primary enhanced image
    enh_pil = TF.to_pil_image(enhanced.squeeze(0).cpu().clamp(0.0, 1.0))
    enh_pil.save(out_dir / f"{base_name}_enhanced.png")

    if save_diagnostics:
        # Illumination map
        if "illumination" in diag and diag["illumination"] is not None:
            l_map = diag["illumination"].squeeze(0).cpu().repeat(3, 1, 1).clamp(0.0, 1.0)
            TF.to_pil_image(l_map).save(out_dir / f"{base_name}_illumination.png")

        # Noise proxy map
        noise_key = "noise_proxy" if "noise_proxy" in diag else "noise_uncertainty"
        if noise_key in diag and diag[noise_key] is not None:
            n_map = diag[noise_key].squeeze(0).cpu().repeat(3, 1, 1).clamp(0.0, 1.0)
            TF.to_pil_image(n_map).save(out_dir / f"{base_name}_noise_proxy.png")

        # Reliability map Q
        if "reliability_map" in diag and diag["reliability_map"] is not None:
            q_map = diag["reliability_map"].squeeze(0).cpu().repeat(3, 1, 1).clamp(0.0, 1.0)
            TF.to_pil_image(q_map).save(out_dir / f"{base_name}_reliability.png")

        # Match confidence map
        if "match_confidence" in diag and diag["match_confidence"] is not None:
            conf_map = diag["match_confidence"].squeeze(0).cpu().repeat(3, 1, 1).clamp(0.0, 1.0)
            TF.to_pil_image(conf_map).save(out_dir / f"{base_name}_match_confidence.png")

        # Absolute difference map
        diff = torch.abs(enhanced - orig).squeeze(0).cpu().clamp(0.0, 1.0)
        TF.to_pil_image(diff).save(out_dir / f"{base_name}_diff.png")

        # Side-by-side comparison: [Original | Enhanced]
        comp = torch.cat([orig.squeeze(0).cpu(), enhanced.squeeze(0).cpu()], dim=2)
        TF.to_pil_image(comp.clamp(0.0, 1.0)).save(out_dir / f"{base_name}_comparison.png")

    print(f"[Done] Saved enhanced image and diagnostics for: {img_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Run LumaGuard-Net Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--diagnostics", action="store_true", help="Save diagnostic intermediate maps")
    parser.add_argument("--max_dim", type=int, default=1200, help="Maximum image dimension for safe inference")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[Inference] Running on device: {device}")

    # Build model using checkpoint config if available
    config = None
    state_dict = None
    model_type = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
            config = ckpt.get("config", None)
            model_type = ckpt.get("model_type", None)
        else:
            state_dict = ckpt

    is_snr = (model_type == "snr_internal_reference") or (
        state_dict is not None
        and any("reference_matcher" in k or "feature_transfer" in k for k in state_dict.keys())
    )

    if is_snr:
        model = build_snr_internal_reference_net(config)
    else:
        model = build_lumaguard_net(config)

    if state_dict is not None:
        model.load_state_dict(state_dict)
        print(f"[Inference] Loaded weights from: {args.checkpoint} ({type(model).__name__})")
    else:
        print("[Inference] Notice: Running with default randomly initialized weights.")


    model.to(device)
    model.eval()

    in_path = Path(args.input)
    out_dir = Path(args.output)

    valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    if in_path.is_file():
        image_files = [in_path]
    elif in_path.is_dir():
        image_files = [
            p for p in in_path.glob("*") if p.suffix.lower() in valid_exts
        ]
    else:
        raise FileNotFoundError(f"Input path not found: {in_path}")

    print(f"[LumaGuard-Net] Processing {len(image_files)} image(s)...")
    for img_p in image_files:
        process_image(
            model,
            img_p,
            out_dir,
            device,
            save_diagnostics=args.diagnostics,
            max_dimension=args.max_dim,
        )

    print(f"[LumaGuard-Net] Inference completed. Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
