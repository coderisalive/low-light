# LumaGuard-Net: Paired RGB U-Net for Low-Light Image Enhancement

This repository currently focuses on **PairedRGBRestorationNet**, a lightweight, paired-supervised RGB U-Net for low-light image enhancement. The default model has **409,347 trainable parameters**.

The model learns to map a low-light RGB image to its paired normal-light target using a multiscale encoder-decoder, skip connections, depthwise-separable residual blocks, and a residual correction in logit-RGB space.

## Repository scope

This codebase keeps one active training/evaluation path for the paired RGB U-Net. Earlier self-supervised, SNR-internal-reference, and ablation experiments have been removed from the current source tree to avoid maintaining multiple competing entry points.

Dataset images and model checkpoints are not stored in Git. Download LOL-v1 from the [dataset's official page](https://daooshee.github.io/BMVC2018website/) and arrange it as follows:

```text
data/lol_dataset/
├── our485/
│   ├── low/
│   └── high/
└── eval15/
    ├── low/
    └── high/
```

Keep the original paired filenames matching between each `low` and `high` directory.

## Install

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Train

From the repository root:

```bash
python -u scripts/train_rgb_restoration.py \
  --data_root data/lol_dataset \
  --epochs 12 \
  --batch_size 4 \
  --crop_size 128 \
  --lr 0.0002
```

The trainer creates a deterministic train/validation split from `our485`. The official `eval15` directory is not loaded during training or checkpoint selection. Early stopping and best-checkpoint saving are enabled.

On Kaggle, set `--data_root` to your local dataset path, for example `/kaggle/working/low-light/data/lol_dataset`.

The trainer writes checkpoints, training history, per-image validation metrics, and labeled comparison images under its experiment output directory. Keep those generated artifacts outside Git.

## Evaluate on eval15

Use the best checkpoint produced by training:

```bash
python -u scripts/evaluate_rgb_restoration.py \
  --checkpoint experiments/paired_rgb_unet_<timestamp>/best_checkpoint.pth \
  --low_dir data/lol_dataset/eval15/low \
  --high_dir data/lol_dataset/eval15/high \
  --output_csv experiments/rgb_unet_eval15.csv \
  --comparison_dir experiments/rgb_unet_eval15_comparisons \
  --device auto
```

Replace the checkpoint path with the actual path printed at the end of training. Evaluation reports PSNR, SSIM, MSE, and MAE and saves a labeled comparison for each matched pair.

## Run inference on images

```bash
python scripts/infer.py \
  --input path/to/low_light_image.png \
  --checkpoint path/to/best_checkpoint.pth \
  --output outputs/enhanced \
  --device auto
```

The input can be one supported image or a directory of images. The script saves an enhanced image and an input/output comparison.

## Metrics and evaluation discipline

- Higher PSNR and SSIM are generally better; lower MSE and MAE are better.
- Compare input and output against the same target images with the same preprocessing.
- Keep `eval15` reserved for final evaluation; do not repeatedly tune model choices against it.
- Report the number of test images alongside aggregate results. LOL-v1's `eval15` split contains only 15 pairs, so results on it are useful but limited.

## Tests

```bash
python -m pytest tests/test_paired_rgb_restoration.py -v
```

## License

MIT. See [LICENSE](LICENSE).
