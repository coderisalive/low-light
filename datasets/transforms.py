"""
Image transforms and augmentations for training and evaluation.
Supports random crops, flips, resizing, and normalization.
"""

from typing import Tuple, Optional
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class PairedRandomCrop:
    """Random crop applied consistently to low-light and reference images."""

    def __init__(self, crop_size: int = 256) -> None:
        self.crop_size = crop_size

    def __call__(
        self, img_low: Image.Image, img_clean: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        w, h = img_low.size
        th, tw = self.crop_size, self.crop_size

        if w < tw or h < th:
            # Resize if smaller than crop size
            img_low = TF.resize(img_low, (max(h, th), max(w, tw)))
            if img_clean is not None:
                img_clean = TF.resize(img_clean, (max(h, th), max(w, tw)))
            w, h = img_low.size

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        img_low_cropped = TF.crop(img_low, i, j, th, tw)
        img_clean_cropped = TF.crop(img_clean, i, j, th, tw) if img_clean is not None else None

        return img_low_cropped, img_clean_cropped


class LowLightAugment:
    """Random horizontal and vertical flips."""

    def __call__(
        self, img_low: Image.Image, img_clean: Optional[Image.Image] = None
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        if random.random() > 0.5:
            img_low = TF.hflip(img_low)
            if img_clean is not None:
                img_clean = TF.hflip(img_clean)

        if random.random() > 0.5:
            img_low = TF.vflip(img_low)
            if img_clean is not None:
                img_clean = TF.vflip(img_clean)

        return img_low, img_clean


def to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    """Converts PIL image (RGB) to float32 Tensor in [0, 1]."""
    return TF.to_tensor(img)
