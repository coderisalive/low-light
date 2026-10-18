"""
Unified Multi-Dataset Paired Loader for Low-Light RGB Restoration.

Supports:
1. LOL-v1 (our485 train, eval15 test)
2. LOL-v2-Real (Real_captured Train, Real_captured Test)
3. LOL-v2-Synthetic (Synthetic Train, Synthetic Test)

Guarantees:
- Pair low/high images correctly and report any unmatched files.
- Collision-proof identifiers (e.g. 'lol_v1:1.png' vs 'lol_v2_real:1.png').
- Deterministic, disjoint train/validation splits per dataset using a configurable seed.
- Official test sets (eval15, LOL-v2-Real Test, LOL-v2-Synthetic Test) are strictly isolated
  from training and checkpoint selection.
- Configurable sampling: proportional, balanced, or custom dataset weights.
- Consistent paired random crops and flips across varied image resolutions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from PIL import Image

from datasets.transforms import PairedRandomCrop, LowLightAugment, to_normalized_tensor


def resolve_subfolder(base: Path, candidate_names: Tuple[str, ...]) -> Optional[Path]:
    """Finds a matching subfolder regardless of capitalization."""
    if not base.is_dir():
        return None
    for name in candidate_names:
        direct = base / name
        if direct.is_dir():
            return direct
    # Case-insensitive check
    for child in base.iterdir():
        if child.is_dir() and child.name.lower() in [c.lower() for c in candidate_names]:
            return child
    return None


def match_low_high_pairs(
    low_dir: Path,
    high_dir: Path,
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> Tuple[List[Tuple[Path, Path, str]], List[Path], List[Path]]:
    """
    Pairs low-light and normal-light image files.
    Matches by exact name, lower-case stem, or normalized stem (stripping 'low_', 'normal_', etc.).
    Returns:
        paired: list of (low_path, high_path, relative_name)
        unmatched_low: list of low files without high match
        unmatched_high: list of high files without low match
    """
    low_files = [f for f in low_dir.glob("*") if f.is_file() and f.suffix.lower() in valid_extensions]
    high_files = [f for f in high_dir.glob("*") if f.is_file() and f.suffix.lower() in valid_extensions]

    high_by_name = {f.name: f for f in high_files}
    high_by_stem = {f.stem.lower(): f for f in high_files}

    def normalize_stem(s: str) -> str:
        s_clean = s.lower()
        for prefix in ("low_", "low", "normal_", "normal", "high_", "high"):
            if s_clean.startswith(prefix):
                s_clean = s_clean[len(prefix):]
        return s_clean.strip("-_ ")

    high_by_norm = {normalize_stem(f.stem): f for f in high_files}

    paired: List[Tuple[Path, Path, str]] = []
    unmatched_low: List[Path] = []
    matched_high_paths = set()

    for l_path in sorted(low_files):
        name = l_path.name
        h_path = None
        if name in high_by_name:
            h_path = high_by_name[name]
        elif l_path.stem.lower() in high_by_stem:
            h_path = high_by_stem[l_path.stem.lower()]
        else:
            norm = normalize_stem(l_path.stem)
            if norm in high_by_norm:
                h_path = high_by_norm[norm]

        if h_path is not None:
            paired.append((l_path, h_path, name))
            matched_high_paths.add(h_path)
        else:
            unmatched_low.append(l_path)

    unmatched_high = [h for h in high_files if h not in matched_high_paths]
    return paired, unmatched_low, unmatched_high


class SinglePairedDataset(Dataset):
    """
    Represents a single paired low/normal dataset split (e.g. LOL-v1 training).
    """

    def __init__(
        self,
        dataset_name: str,
        low_dir: Union[str, Path],
        high_dir: Union[str, Path],
        crop_size: Optional[int] = None,
        augment: bool = False,
        valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.valid_extensions = valid_extensions

        self.crop_fn = PairedRandomCrop(crop_size) if crop_size else None
        self.aug_fn = LowLightAugment() if augment else None

        if not self.low_dir.is_dir() or not self.high_dir.is_dir():
            raise FileNotFoundError(
                f"[{dataset_name}] Folders not found: low='{self.low_dir}', high='{self.high_dir}'"
            )

        self.paired_files, self.unmatched_low, self.unmatched_high = match_low_high_pairs(
            self.low_dir, self.high_dir, self.valid_extensions
        )

        if len(self.paired_files) == 0:
            raise RuntimeError(
                f"[{dataset_name}] Found 0 matched pairs in low='{self.low_dir}' and high='{self.high_dir}'"
            )

    def __len__(self) -> int:
        return len(self.paired_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Returns:
            low_t: (3, H, W) in [0, 1]
            high_t: (3, H, W) in [0, 1]
            name: raw filename
            unique_id: collision-proof identifier 'dataset_name:filename'
        """
        low_p, high_p, name = self.paired_files[idx]
        with Image.open(low_p) as l_img, Image.open(high_p) as h_img:
            l_img = l_img.convert("RGB")
            h_img = h_img.convert("RGB")

        if self.crop_fn is not None:
            l_img, h_img = self.crop_fn(l_img, h_img)

        if self.aug_fn is not None:
            l_img, h_img = self.aug_fn(l_img, h_img)

        low_t = to_normalized_tensor(l_img)
        high_t = to_normalized_tensor(h_img)
        unique_id = f"{self.dataset_name}:{name}"
        return low_t, high_t, name, unique_id


class MultiDatasetPairedDataset(Dataset):
    """
    Combines multiple SinglePairedDataset splits with collision-proof identifiers
    and balanced/weighted sampling support.
    """

    def __init__(
        self,
        datasets: List[SinglePairedDataset],
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.dataset_offsets: List[int] = []
        self.total_length = 0

        for d in datasets:
            self.dataset_offsets.append(self.total_length)
            self.total_length += len(d)

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        # Binary or linear search for sub-dataset
        for i in range(len(self.datasets) - 1, -1, -1):
            if idx >= self.dataset_offsets[i]:
                return self.datasets[i][idx - self.dataset_offsets[i]]
        raise IndexError(f"Index {idx} out of range for MultiDataset of size {self.total_length}")

    def get_sampling_weights(
        self,
        mode: str = "proportional",
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Computes sample-level weights for WeightedRandomSampler:
        - 'proportional': uniform weight per image (1.0).
        - 'balanced': equal total probability mass for each dataset (1.0 / len(ds) per sample).
        - 'weighted': custom_weights[dataset_name] distributed across each dataset's samples.
        """
        weights = []
        num_datasets = len(self.datasets)
        for d in self.datasets:
            n_samples = len(d)
            if mode == "balanced":
                w = 1.0 / (n_samples * num_datasets)
            elif mode == "weighted" and custom_weights and d.dataset_name in custom_weights:
                user_w = custom_weights[d.dataset_name]
                w = float(user_w) / n_samples
            else:
                w = 1.0
            weights.extend([w] * n_samples)
        return torch.tensor(weights, dtype=torch.double)


def discover_lol_v1_paths(root: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Discovers (train_low, train_high, test_low, test_high) for LOL-v1."""
    train_low = root / "our485" / "low"
    train_high = root / "our485" / "high"
    test_low = root / "eval15" / "low"
    test_high = root / "eval15" / "high"

    if train_low.is_dir() and train_high.is_dir():
        return (
            train_low,
            train_high,
            test_low if test_low.is_dir() else None,
            test_high if test_high.is_dir() else None,
        )

    # Search subdirectories
    for cand in root.rglob("our485"):
        tl = cand / "low"
        th = cand / "high"
        if tl.is_dir() and th.is_dir():
            base = cand.parent
            el = base / "eval15" / "low"
            eh = base / "eval15" / "high"
            return (tl, th, el if el.is_dir() else None, eh if eh.is_dir() else None)

    return None, None, None, None


def discover_lol_v2_real_paths(root: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Discovers (train_low, train_high, test_low, test_high) for LOL-v2-Real."""
    base = root
    real_cand = resolve_subfolder(root, ("Real_captured", "Real", "real_captured", "real", "_hf_raw"))
    if real_cand is not None:
        base = real_cand
        nested = resolve_subfolder(real_cand, ("_hf_raw",))
        if nested is not None:
            base = nested

    train_folder = resolve_subfolder(base, ("Train", "train"))
    test_folder = resolve_subfolder(base, ("Test", "test"))

    tl, th, el, eh = None, None, None, None
    if train_folder is not None:
        tl = resolve_subfolder(train_folder, ("Low", "low", "Input", "input"))
        th = resolve_subfolder(train_folder, ("Normal", "normal", "high", "High", "GT", "gt"))

    if test_folder is not None:
        el = resolve_subfolder(test_folder, ("Low", "low", "Input", "input"))
        eh = resolve_subfolder(test_folder, ("Normal", "normal", "high", "High", "GT", "gt"))

    return tl, th, el, eh


def discover_lol_v2_synthetic_paths(root: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Discovers (train_low, train_high, test_low, test_high) for LOL-v2-Synthetic."""
    base = root
    syn_cand = resolve_subfolder(root, ("Synthetic", "synthetic", "Syn", "syn", "_hf_raw"))
    if syn_cand is not None:
        base = syn_cand
        nested = resolve_subfolder(syn_cand, ("_hf_raw",))
        if nested is not None:
            base = nested

    train_folder = resolve_subfolder(base, ("Train", "train"))
    test_folder = resolve_subfolder(base, ("Test", "test"))

    tl, th, el, eh = None, None, None, None
    if train_folder is not None:
        tl = resolve_subfolder(train_folder, ("Low", "low", "Input", "input"))
        th = resolve_subfolder(train_folder, ("Normal", "normal", "high", "High", "GT", "gt"))

    if test_folder is not None:
        el = resolve_subfolder(test_folder, ("Low", "low", "Input", "input"))
        eh = resolve_subfolder(test_folder, ("Normal", "normal", "high", "High", "GT", "gt"))

    return tl, th, el, eh


def build_multi_paired_train_val(
    dataset_roots: Dict[str, Union[str, Path]],
    crop_size: int = 192,
    validation_fraction: float = 0.10,
    seed: int = 42,
    augment: bool = True,
) -> Tuple[MultiDatasetPairedDataset, Dict[str, SinglePairedDataset], Dict[str, Dict[str, int]]]:
    """
    Builds pooled training dataset and per-dataset validation datasets.
    Validation splits are strictly carved out of training sets using a deterministic random seed.
    Official test sets are NEVER included in training or validation splits.

    Returns:
        train_combined: MultiDatasetPairedDataset for training
        val_datasets_dict: dict mapping dataset_name -> SinglePairedDataset (validation)
        summary_stats: dict of train/val pair counts per dataset
    """
    train_splits: List[SinglePairedDataset] = []
    val_splits: Dict[str, SinglePairedDataset] = {}
    stats: Dict[str, Dict[str, int]] = {}

    rng = random.Random(seed)

    for name, root_path in dataset_roots.items():
        if root_path is None:
            continue
        root = Path(root_path)
        if not root.is_dir():
            continue

        tl, th, el, eh = None, None, None, None
        name_lower = name.lower()
        if "lol_v1" in name_lower or "lolv1" in name_lower:
            tl, th, el, eh = discover_lol_v1_paths(root)
            clean_name = "lol_v1"
        elif "real" in name_lower:
            tl, th, el, eh = discover_lol_v2_real_paths(root)
            clean_name = "lol_v2_real"
        elif "synthetic" in name_lower:
            tl, th, el, eh = discover_lol_v2_synthetic_paths(root)
            clean_name = "lol_v2_synthetic"
        else:
            # Generic attempt
            tl = resolve_subfolder(root, ("low", "Low"))
            th = resolve_subfolder(root, ("high", "High", "normal", "Normal"))
            clean_name = name_lower

        if tl is None or th is None or not (tl.is_dir() and th.is_dir()):
            print(f"[MultiDataset Warning] Could not resolve training low/high folders in {root}")
            continue

        # Full training pairs discovered
        try:
            full_train = SinglePairedDataset(clean_name, tl, th, crop_size=None, augment=False)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"[MultiDataset Warning] Skipping '{clean_name}': {e}")
            continue
        n_total = len(full_train)
        if n_total < 2:
            print(f"[MultiDataset Warning] Dataset {clean_name} has fewer than 2 matched pairs. Skipping.")
            continue

        # Deterministic split per dataset
        indices = list(range(n_total))
        rng.shuffle(indices)
        n_val = max(1, min(n_total - 1, round(n_total * validation_fraction)))
        val_idx = sorted(indices[:n_val])
        train_idx = sorted(indices[n_val:])

        # Build train subset with augmentations and crops
        ds_train = SinglePairedDataset(clean_name, tl, th, crop_size=crop_size, augment=augment)
        ds_train.paired_files = [full_train.paired_files[i] for i in train_idx]
        train_splits.append(ds_train)

        # Build val subset without crops or augmentations
        ds_val = SinglePairedDataset(clean_name, tl, th, crop_size=None, augment=False)
        ds_val.paired_files = [full_train.paired_files[i] for i in val_idx]
        val_splits[clean_name] = ds_val

        stats[clean_name] = {
            "total_available": n_total,
            "train_count": len(ds_train),
            "val_count": len(ds_val),
            "unmatched_low": len(full_train.unmatched_low),
            "unmatched_high": len(full_train.unmatched_high),
        }

    if not train_splits:
        raise RuntimeError("No valid paired datasets could be loaded from the provided roots.")

    train_combined = MultiDatasetPairedDataset(train_splits)
    return train_combined, val_splits, stats


def build_official_test_datasets(
    dataset_roots: Dict[str, Union[str, Path]],
) -> Dict[str, SinglePairedDataset]:
    """
    Builds official test datasets (e.g. eval15, LOL-v2-Real Test, LOL-v2-Synthetic Test).
    Guarantees that test sets are completely isolated from train/val splits.
    """
    test_datasets: Dict[str, SinglePairedDataset] = {}

    for name, root_path in dataset_roots.items():
        if root_path is None:
            continue
        root = Path(root_path)
        if not root.is_dir():
            continue

        tl, th, el, eh = None, None, None, None
        name_lower = name.lower()
        if "lol_v1" in name_lower or "lolv1" in name_lower:
            tl, th, el, eh = discover_lol_v1_paths(root)
            clean_name = "lol_v1"
        elif "real" in name_lower:
            tl, th, el, eh = discover_lol_v2_real_paths(root)
            clean_name = "lol_v2_real"
        elif "synthetic" in name_lower:
            tl, th, el, eh = discover_lol_v2_synthetic_paths(root)
            clean_name = "lol_v2_synthetic"
        else:
            clean_name = name_lower

        if el is not None and eh is not None and el.is_dir() and eh.is_dir():
            ds_test = SinglePairedDataset(clean_name, el, eh, crop_size=None, augment=False)
            test_datasets[clean_name] = ds_test
            print(f"[Test Dataset] {clean_name}: {len(ds_test)} pairs loaded from '{el}' and '{eh}'")

    return test_datasets
