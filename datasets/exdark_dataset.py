"""
ExDark (Exclusively Dark Image Dataset) Loader.

Official Source:
    https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

Properties:
- 7,363 low-light images across 12 object classes:
  Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table.
- STRICTLY UNPAIRED: Contains real-world nighttime images with NO normal-light references.
- Normal-light references are NEVER fabricated.
- Supports official Groundtruth/imageclasslist.txt split annotations when provided,
  or a deterministic seed/hash split as a reliable fallback.
- Never passed into paired full-reference loss functions.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import os
import hashlib
import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets.transforms import PairedRandomCrop, LowLightAugment, to_normalized_tensor


class ExDarkDataset(Dataset):
    """
    Unpaired low-light dataset loader for ExDark.
    """

    CLASSES = [
        "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
        "Chair", "Cup", "Dog", "Motorbike", "People", "Table"
    ]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        crop_size: Optional[int] = 192,
        augment: bool = True,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
        valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        if self.split not in ("train", "val", "test", "all"):
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', 'test', or 'all'.")

        self.crop_size = crop_size
        self.augment = augment and (self.split == "train")
        self.valid_extensions = valid_extensions
        self.seed = seed

        self.crop_fn = PairedRandomCrop(crop_size) if (crop_size and self.split == "train") else None
        self.aug_fn = LowLightAugment() if self.augment else None

        self.samples: List[Tuple[Path, str, str]] = []  # (path, category, identifier)

        if self.root_dir.is_dir():
            self._load_samples(val_ratio, test_ratio)

    def _find_annotation_file(self) -> Optional[Path]:
        candidates = [
            self.root_dir / "Groundtruth" / "imageclasslist.txt",
            self.root_dir / "imageclasslist.txt",
            self.root_dir / "Groundtruth" / "imageclasses.txt",
            self.root_dir / "imageclasses.txt",
        ]
        for c in candidates:
            if c.is_file():
                return c
        # Search one level deep
        for c in self.root_dir.glob("**/imageclass*.txt"):
            if c.is_file():
                return c
        return None

    def _load_samples(self, val_ratio: float, test_ratio: float) -> None:
        # 1. Discover all class-associated image files
        file_map: Dict[str, Tuple[Path, str]] = {}
        for root, _, files in os.walk(self.root_dir):
            root_p = Path(root)
            # Avoid Groundtruth directory for images
            if "groundtruth" in root_p.name.lower():
                continue
            for f in files:
                if f.lower().endswith(self.valid_extensions):
                    p = root_p / f
                    cat = root_p.name
                    file_map[f] = (p, cat)
                    file_map[f.lower()] = (p, cat)

        # 2. Check for official imageclasslist.txt
        ann_file = self._find_annotation_file()
        used_official_split = False

        if ann_file is not None:
            try:
                with open(ann_file, "r", encoding="utf-8", errors="ignore") as stream:
                    lines = [line.strip().split() for line in stream if line.strip()]

                for parts in lines:
                    if len(parts) >= 2:
                        fname = parts[0]
                        cat = parts[1] if len(parts) >= 3 else parts[0].split("/")[0] if "/" in parts[0] else "Unknown"
                        split_code = parts[2] if len(parts) >= 3 else parts[1]

                        # Code interpretation:
                        # 1 = Train, 2 = Val, 3 = Test, or string matches
                        is_target = False
                        code_str = str(split_code).lower()
                        if self.split == "all":
                            is_target = True
                        elif self.split == "train" and (code_str in ("1", "train", "training")):
                            is_target = True
                        elif self.split == "val" and (code_str in ("2", "val", "validation")):
                            is_target = True
                        elif self.split == "test" and (code_str in ("3", "test", "testing")):
                            is_target = True

                        if is_target:
                            basename = Path(fname).name
                            resolved = file_map.get(basename) or file_map.get(basename.lower())
                            if resolved is not None:
                                p, detected_cat = resolved
                                final_cat = cat if cat != "Unknown" else detected_cat
                                ident = f"exdark:{final_cat}/{p.name}"
                                self.samples.append((p, final_cat, ident))
                                used_official_split = True
            except Exception:
                used_official_split = False

        # 3. Fallback: Deterministic seed/hash split if official annotation is not provided or incomplete
        if not used_official_split or len(self.samples) == 0:
            self.samples.clear()
            unique_paths = sorted(list({p for p, _ in file_map.values()}), key=lambda x: str(x))
            for p in unique_paths:
                cat = p.parent.name
                # Deterministic MD5 hash to guarantee platform-independent split
                h = int(hashlib.md5(f"{self.seed}_{p.name}".encode("utf-8")).hexdigest(), 16) % 10000
                frac = h / 10000.0

                if self.split == "all":
                    match = True
                elif self.split == "test":
                    match = (frac >= (1.0 - test_ratio))
                elif self.split == "val":
                    match = ((1.0 - test_ratio - val_ratio) <= frac < (1.0 - test_ratio))
                elif self.split == "train":
                    match = (frac < (1.0 - test_ratio - val_ratio))
                else:
                    match = False

                if match:
                    ident = f"exdark:{cat}/{p.name}"
                    self.samples.append((p, cat, ident))

        self.samples.sort(key=lambda x: str(x[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str, str]:
        """
        Returns:
            tensor_img: (3, H, W) normalized tensor in [0, 1]
            category: class string (e.g. 'Bicycle', 'Car')
            filename: basename of the file
            identifier: collision-proof identifier 'exdark:<category>/<filename>'
        """
        path, category, ident = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.crop_fn is not None:
            img, _ = self.crop_fn(img)

        if self.aug_fn is not None:
            img, _ = self.aug_fn(img)

        t_img = to_normalized_tensor(img)
        return t_img, category, path.name, ident
