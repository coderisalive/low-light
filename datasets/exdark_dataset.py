"""
ExDark Dataset Loader.
Supports real-world low-light images from ExDark dataset without synthetic ground-truth fabrication.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets.transforms import PairedRandomCrop, to_normalized_tensor


class ExDarkDataset(Dataset):
    """
    ExDark (Exclusively Dark Image Dataset) loader.
    Does NOT fabricate normal-light references. Exclusively for real-world robustness.
    """

    def __init__(
        self,
        root_dir: str,
        crop_size: Optional[int] = None,
        valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.crop_fn = PairedRandomCrop(crop_size) if crop_size else None
        self.valid_extensions = valid_extensions

        self.samples: List[Tuple[Path, str]] = []
        if self.root_dir.exists():
            for root, _, files in os.walk(self.root_dir):
                for f in files:
                    if f.lower().endswith(self.valid_extensions):
                        p = Path(root) / f
                        category = p.parent.name
                        self.samples.append((p, category))
            self.samples.sort()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        path, category = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.crop_fn is not None:
            img, _ = self.crop_fn(img)

        t_img = to_normalized_tensor(img)
        return t_img, category, path.name
