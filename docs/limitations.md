# Limitations

- **Small paired dataset:** training uses 485 LOL-v1 pairs; generalization to other cameras, lighting conditions, and image types is not guaranteed.
- **Small final test split:** `eval15` has 15 pairs. Results should be described as a benchmark on this split, not proof of broad real-world performance.
- **Supervised dependency:** the active RGB U-Net needs aligned low-/normal-light image pairs for training. It is not a self-supervised or unpaired method.
- **Restoration trade-offs:** higher brightness or PSNR does not guarantee natural color, realistic texture, or preservation of fine details. Inspect output images as well as metrics.
- **Metric limitations:** PSNR/MAE measure pixel errors, while the repository's SSIM implementation is a simple reference metric. Report metric implementation and preprocessing when comparing to published results.
- **No cross-dataset claim:** evaluate on additional datasets and camera conditions before making general robustness claims.
