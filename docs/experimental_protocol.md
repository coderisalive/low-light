# Experimental Protocol

## Dataset layout

Use the paired LOL-v1 data in the following layout:

- `our485/low`: low-light training inputs
- `our485/high`: paired normal-light targets
- `eval15/low`: held-out test inputs
- `eval15/high`: corresponding test targets

Paired filenames must match. The dataset itself is not tracked in this repository.

## Train/validation split

The RGB U-Net trainer makes a deterministic split from `our485` using the configured seed and validation fraction (default 10%). Training and checkpoint selection use only this split. The `eval15` directory is not loaded by the trainer.

## Final evaluation

Run `scripts/evaluate_rgb_restoration.py` once the checkpoint is selected. The script reports PSNR, SSIM, MSE, and MAE and saves per-image results plus labeled input/output/target/error comparisons.

When making claims, report:

- the dataset and split;
- number of evaluated images;
- model checkpoint/configuration;
- mean metrics and, ideally, per-image metrics;
- whether input and output use exactly the same preprocessing.

LOL-v1's `eval15` split contains 15 pairs, so avoid broad generalization from this test alone. Do not use `eval15` to select hyperparameters or choose among checkpoints.
