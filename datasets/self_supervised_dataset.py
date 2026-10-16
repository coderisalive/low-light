"""
Self-Supervised Low-Light Dataset.
Crucial Scientific Constraint:
In self-supervised training mode, this dataset EXPOSES LOW-LIGHT IMAGES ONLY.
It NEVER loads or consumes normal-light target images.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets.transforms import PairedRandomCrop, LowLightAugment, to_normalized_tensor


class SelfSupervisedLowLightDataset(Dataset):
    """
    Self-supervised dataset loader.
    Only loads unpaired low-light input images.
    """

    def __init__(
        self,
        image_dir: str,
        crop_size: Optional[int] = 256,
        augment: bool = True,
        valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.valid_extensions = valid_extensions

        self.crop_fn = PairedRandomCrop(crop_size) if crop_size else None
        self.aug_fn = LowLightAugment() if augment else None

        self.image_paths: List[Path] = []
        if self.image_dir.exists():
            for root, _, files in os.walk(self.image_dir):
                for f in files:
                    if f.lower().endswith(self.valid_extensions):
                        self.image_paths.append(Path(root) / f)
            self.image_paths.sort()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception as e:
            # Fallback or skip corrupted image
            raise RuntimeError(f"Failed to read image {path}: {e}")

        if self.crop_fn is not None:
            img, _ = self.crop_fn(img)
        if self.aug_fn is not None:
            img, _ = self.aug_fn(img)

        tensor_img = to_normalized_tensor(img)
        return tensor_img
