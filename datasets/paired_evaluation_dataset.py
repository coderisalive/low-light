"""
Paired Evaluation Dataset.
Used STRICTLY during validation and testing to compute reference metrics (PSNR, SSIM, LPIPS).
Never used by the self-supervised training loss.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets.transforms import PairedRandomCrop, to_normalized_tensor


class PairedEvaluationDataset(Dataset):
    """
    Paired dataset loader for LOL-v1, LOL-v2, etc.
    Matches low-light images with corresponding high-light ground truths.
    """

    def __init__(
        self,
        low_dir: str,
        high_dir: str,
        crop_size: Optional[int] = None,
        valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
        filenames: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.crop_size = crop_size
        self.valid_extensions = valid_extensions
        self.crop_fn = PairedRandomCrop(crop_size) if crop_size else None

        self.paired_files: List[Tuple[Path, Path, str]] = []
        selected_names = set(filenames) if filenames is not None else None

        if self.low_dir.exists() and self.high_dir.exists():
            low_files = {
                f.name: f
                for f in self.low_dir.glob("*")
                if f.suffix.lower() in self.valid_extensions
                and (selected_names is None or f.name in selected_names)
            }
            high_files = {
                f.name: f
                for f in self.high_dir.glob("*")
                if f.suffix.lower() in self.valid_extensions
                and (selected_names is None or f.name in selected_names)
            }

            for name, low_p in sorted(low_files.items()):
                if name in high_files:
                    self.paired_files.append((low_p, high_files[name], name))

    def __len__(self) -> int:
        return len(self.paired_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        low_p, high_p, name = self.paired_files[idx]
        with Image.open(low_p) as l_img, Image.open(high_p) as h_img:
            l_img = l_img.convert("RGB")
            h_img = h_img.convert("RGB")

        if self.crop_fn is not None:
            l_img, h_img = self.crop_fn(l_img, h_img)

        t_low = to_normalized_tensor(l_img)
        t_high = to_normalized_tensor(h_img)

        return t_low, t_high, name
