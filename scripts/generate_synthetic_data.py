"""
Generates synthetic low-light / normal-light paired samples for smoke tests and pipeline verification.
"""

from pathlib import Path
import numpy as np
from PIL import Image


def create_synthetic_datasets(base_dir: str = "data/synthetic_sample") -> None:
    base = Path(base_dir)
    train_low = base / "train" / "low"
    val_low = base / "val" / "low"
    val_high = base / "val" / "high"

    for d in [train_low, val_low, val_high]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    # 4 training low-light samples
    for i in range(4):
        # Create dark pattern with gradient
        arr = rng.randint(5, 45, size=(128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(train_low / f"synthetic_train_{i:02d}.png")

    # 2 validation paired samples
    for i in range(2):
        arr_high = rng.randint(120, 240, size=(128, 128, 3), dtype=np.uint8)
        # Low light version: attenuated + Poisson/Gaussian noise
        arr_low = (arr_high.astype(np.float32) * 0.15 + rng.normal(0, 5, size=(128, 128, 3))).clip(0, 255).astype(np.uint8)

        img_h = Image.fromarray(arr_high)
        img_l = Image.fromarray(arr_low)

        img_h.save(val_high / f"synthetic_val_{i:02d}.png")
        img_l.save(val_low / f"synthetic_val_{i:02d}.png")

    print(f"Generated synthetic verification data at: {base}")


if __name__ == "__main__":
    create_synthetic_datasets()
