# LumaGuard-Net: Paired RGB U-Net for Low-Light Image Enhancement

This repository implements **PairedRGBRestorationNet**, a lightweight, paired-supervised RGB U-Net for low-light image enhancement. The default model has **409,347 trainable parameters**.

The model learns to map low-light RGB images to their paired normal-light targets using a multiscale encoder-decoder, skip connections, depthwise-separable residual blocks, and a residual correction in logit-RGB space.

---

## Supported Datasets

The repository supports multi-dataset training and multi-benchmark testing across four standard low-light datasets:

| Dataset | Type | Training Pairs / Images | Test Pairs / Images | Official Source / Citation |
| :--- | :---: | :---: | :---: | :--- |
| **LOL-v1** | Paired | 485 (`our485`) | 15 (`eval15`) | [Wei et al., BMVC 2018](https://daooshee.github.io/BMVC2018website/) |
| **LOL-v2-Real** | Paired | 689 (`Train`) | 100 (`Test`) | [Yang et al., IJCV 2021](https://github.com/flyywh/LOL-v2) |
| **LOL-v2-Synthetic** | Paired | 900 (`Train`) | 100 (`Test`) | [Yang et al., IJCV 2021](https://github.com/flyywh/LOL-v2) |
| **ExDark** | **Unpaired** | 5,890 (train split) | 1,473 (test split) | [Loh & Chan, CVIU 2019](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) |

> [!IMPORTANT]
> **Paired vs. Unpaired Distinction**:
> - **LOL-v1, LOL-v2-Real, and LOL-v2-Synthetic** contain matched low-/normal-light pairs. They are used for paired supervised training and full-reference evaluation (**PSNR, SSIM, MSE, MAE**).
> - **ExDark** contains real-world nighttime low-light images across 12 object classes (**no normal-light ground truth**). It is evaluated strictly via no-reference metrics (**Information Entropy, RMS Contrast, visual inspection**). Normal-light references are **never fabricated**, and ExDark is never mixed into paired full-reference loss functions.

---

## Dataset Directory Layouts

Datasets should be stored outside the Git repository (e.g. under `data/` or a custom path).

### 1. LOL-v1
```text
data/lol_v1/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

### 2. LOL-v2-Real
```text
data/LOL-v2-Real/
└── Real_captured/
    ├── Train/
    │   ├── Low/
    │   └── Normal/
    └── Test/
        ├── Low/
        └── Normal/
```

### 3. LOL-v2-Synthetic
```text
data/LOL-v2-Synthetic/
└── Synthetic/
    ├── Train/
    │   ├── Low/
    │   └── Normal/
    └── Test/
        ├── Low/
        └── Normal/
```

### 4. ExDark
```text
data/ExDark/
├── Groundtruth/
│   └── imageclasslist.txt      # (Optional official train/val/test split)
├── Bicycle/
├── Boat/
├── Bottle/
├── Bus/
├── Car/
├── Cat/
├── Chair/
├── Cup/
├── Dog/
├── Motorbike/
├── People/
└── Table/
```

---

## Installation

Python 3.10+ and PyTorch 2.0+ are recommended.

```bash
pip install -r requirements.txt
```

---

## Downloading Datasets

A turnkey downloader script is included to automatically fetch and organize **LOL-v2 Real**, **LOL-v2 Synthetic**, and **ExDark**:

```bash
# Install Hugging Face datasets library (if not already installed)
pip install datasets huggingface_hub

# Download LOL-v2 Real & Synthetic and ExDark into data/
python scripts/download_lolv2.py --output_root data/ --datasets all
```

On Kaggle, you can run:
```bash
python scripts/download_lolv2.py --output_root /kaggle/working/datasets/ --datasets all
```

---

## Training

### Multi-Dataset Training (LOL-v1 + LOL-v2-Real + LOL-v2-Synthetic)
Train on all three paired datasets simultaneously with balanced sampling:

```bash
python -u scripts/train_rgb_restoration.py \
  --lolv1_root data/lol_v1 \
  --lolv2_real_root data/LOL-v2-Real \
  --lolv2_synthetic_root data/LOL-v2-Synthetic \
  --sampling_mode balanced \
  --epochs 25 \
  --batch_size 8 \
  --crop_size 192 \
  --lr 0.0002 \
  --seed 42
```

### Single Dataset Training (LOL-v1 only)
```bash
python -u scripts/train_rgb_restoration.py \
  --lolv1_root data/lol_dataset \
  --epochs 12 \
  --batch_size 4 \
  --crop_size 128 \
  --lr 0.0002
```

### Sampling Strategies
- `--sampling_mode balanced` (default): Balances dataset weights so each enabled dataset contributes equally in expectation.
- `--sampling_mode proportional`: Samples uniformly from all available images in proportion to dataset sizes.
- `--sampling_mode weighted --dataset_weights lol_v1:1.0,lol_v2_real:1.5,lol_v2_synthetic:0.5`: Applies custom sampling multipliers.

### Full Epoch Training (Disabling Early Stopping)
Agar aapko pure epochs bina ruke train karna hai (bina early stopping ke), to `--no_early_stopping` flag pass karein:
```bash
python -u scripts/train_rgb_restoration.py \
  --lolv1_root /kaggle/working/lol_compat \
  --lolv2_real_root /kaggle/input/lolv2-dataset/LOL-v2-Real \
  --lolv2_synthetic_root /kaggle/input/lolv2-dataset/LOL-v2-Synthetic \
  --epochs 35 \
  --no_early_stopping
```

### Data Leakage Prevention Rules
- The trainer carves a deterministic validation split (default 10%) out of the training pools only.
- Official test sets (`eval15`, `LOL-v2-Real Test`, `LOL-v2-Synthetic Test`) are strictly isolated and never loaded during training or checkpoint selection.

---

## Evaluation

### Multi-Benchmark Evaluation
Evaluate a trained checkpoint across all available official test sets in one command:

```bash
python -u scripts/evaluate_rgb_restoration.py \
  --checkpoint experiments/paired_rgb_unet_<timestamp>/best_checkpoint.pth \
  --lolv1_root data/lol_v1 \
  --lolv2_real_root data/LOL-v2-Real \
  --lolv2_synthetic_root data/LOL-v2-Synthetic \
  --exdark_root data/ExDark \
  --output_dir experiments/multi_benchmark_results
```

This outputs:
1. `benchmark_summary.csv`: A leaderboard report with PSNR, SSIM, and MAE for paired benchmarks, and Entropy & RMS Contrast for ExDark.
2. Individual per-image CSVs: `lol_v1_test_per_image.csv`, `lol_v2_real_test_per_image.csv`, `lol_v2_synthetic_test_per_image.csv`, `exdark_unpaired_metrics.csv`.
3. 4-panel visual comparison figures: `[Low-light Input | RGB U-Net Output | Ground Truth | Residual x4]`.

### Single Benchmark Evaluation
```bash
python -u scripts/evaluate_rgb_restoration.py \
  --checkpoint path/to/best_checkpoint.pth \
  --low_dir data/lol_dataset/eval15/low \
  --high_dir data/lol_dataset/eval15/high \
  --output_dir experiments/eval15_results
```

---

## Inference on Unpaired Images

To enhance arbitrary low-light images or directory of images:

```bash
python scripts/infer.py \
  --input path/to/image_or_folder \
  --checkpoint path/to/best_checkpoint.pth \
  --output outputs/enhanced \
  --device auto
```

---

## Automated Tests

Run the full automated test suite (including synthetic folder layout fixtures, filename pairing, split isolation, and ExDark parsing):

```bash
python -m pytest tests/ -v
```

---

## Citations and Licenses

- **LOL-v1**: Wei et al., *"Deep Retinex Decomposition for Low-Light Enhancement"*, BMVC 2018. Available for non-commercial research.
- **LOL-v2**: Yang et al., *"Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement"*, IJCV 2021.
- **ExDark**: Loh & Chan, *"Getting to know low-light images with the Exclusively Dark dataset"*, CVIU 2019. Available for academic research.
- **Codebase License**: MIT. See [LICENSE](LICENSE).
