# Experimental Protocol & Evaluation Guidelines

## 1. Datasets and Setup

### A. LOL-v1 (BMVC 2018)
- **Source**: [https://daooshee.github.io/BMVC2018website/](https://daooshee.github.io/BMVC2018website/)
- **Protocol**:
  - Training: 485 low-light training images.
  - Crucial constraint: Self-supervised training only accesses the `our485/low` directory. Clean normal-light pairs (`our485/high`) are **strictly prohibited** from loss computation.
  - Evaluation: 15 official test pairs (`eval15/low` and `eval15/high`).

### B. LOL-v2 (CVPR 2022)
- **Source**: [CVPR 2022 SNR-Aware Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.html)
- Separate evaluation tracks for Real and Synthetic subsets.

### C. ExDark (Exclusively Dark Dataset)
- **Source**: [https://github.com/cs-chan/Exclusively-Dark-Image-Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- Real-world unpaired images across 12 semantic classes.
- Used strictly for no-reference evaluation (Entropy, Contrast) and visual verification. No synthetic normal-light ground truths are fabricated.

## 2. Evaluation Metrics Protocol
- **Full-Reference (LOL-v1 & LOL-v2)**:
  - Peak Signal-to-Noise Ratio (PSNR in dB)
  - Structural Similarity Index (SSIM)
  - Mean Squared Error (MSE) & Mean Absolute Error (MAE)
  - Optional Learned Perceptual Image Patch Similarity (LPIPS)
- **No-Reference (ExDark & in-the-wild low-light captures)**:
  - Shannon Entropy of luminance
  - RMS Contrast
- **Efficiency & Footprint**:
  - Trainable parameter count
  - GFLOPs at fixed $400 \times 600$ input resolution
  - Latency (ms) and Throughput (FPS) with warmup runs

## 3. Baselines for Comparison
Published reference models:
- Zero-DCE (CVPR 2020)
- RetinexNet (BMVC 2018)
- EnlightenGAN (IEEE TIP 2021)
- SCI (CVPR 2022)
- SNR-Aware Net (CVPR 2022)
- ZERO-IG / ZLEN

All benchmark numbers must be gathered under identical resolutions and hardware environments without fabricating values.
