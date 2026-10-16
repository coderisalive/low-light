# LumaGuard-Net: Lightweight Noise-Guided Self-Supervised Network for Low-Light Image Enhancement

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Parameter Budget](https://img.shields.io/badge/Parameters-27.0k%20%28%3C1M%20Budget%29-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Scientific Premise**: LumaGuard-Net investigates whether explicitly estimating spatial illumination and noise uncertainty proxies within a depthwise-separable convolutional architecture can improve the quality-efficiency trade-off in reference-free (self-supervised) low-light image enhancement.

---

## Key Features

1. **Self-Supervised Optimization**:
   - Zero requirement for paired normal-light ground truths during training.
   - Clean targets are strictly isolated for validation and testing metrics.
2. **Four-Module Architecture**:
   - **Module A (INE)**: Illumination-Noise Estimator predicting illumination map $L \in (0, 1]$ and noise uncertainty proxy $N \in [0, 1]$.
   - **Module B**: Multi-scale depthwise-separable residual extractor with dilated receptive fields.
   - **Module C**: Reliability-guided feature modulation with learnable reliability gate $Q = \sigma(a \log(L+\epsilon) - b N + c)$.
   - **Module D**: Bounded residual luminance/chrominance head preventing overexposure and color casts.
3. **Extreme Efficiency**:
   - Total trainable parameters: **27,049** (well under the 1,000,000 parameter budget).
   - Estimated FLOPs: **9.00 GFLOPs** at $400 \times 600$ resolution.
4. **Comprehensive Research Suite**:
   - Unit tests covering model shapes, gradient flow, and parameter budgets.
   - Full ablation matrix script comparing noise estimation, reliability gates, separable vs standard convolutions, and loss terms.
   - Rich diagnostic visual exports ($I_{\text{low}}, I_{\text{enh}}, L, N, Q, \Delta Y, \Delta C$, difference map, and comparisons).

---

## Repository Structure

```text
LumaGuard-Net/
├── configs/
│   ├── default.yaml            # Default full-scale training configuration
│   └── smoke_test.yaml         # Fast smoke test configuration
├── datasets/
│   ├── self_supervised_dataset.py  # Strictly low-light loader (no reference access)
│   ├── paired_evaluation_dataset.py # Validation/test loader with ground truth
│   ├── exdark_dataset.py       # Real-world unpaired ExDark loader
│   └── transforms.py           # Augmentations and tensor conversions
├── models/
│   ├── illumination_noise_estimator.py # Module A: Illumination & Noise Estimator
│   ├── lightweight_blocks.py   # Module B: Depthwise-separable residual blocks
│   ├── reliability_gate.py     # Module C: Reliability modulation & gating
│   ├── enhancement_head.py     # Module D: Bounded residual YCbCr/RGB head
│   └── lumaguard_net.py        # Top-level model assembling all modules
├── losses/
│   ├── masked_consistency.py   # Blind-spot masked prediction loss
│   ├── exposure_loss.py        # Patch-based exposure control loss
│   ├── illumination_smoothness.py # Edge-aware gradient smoothness
│   ├── color_consistency.py    # Patch-wise chromatic vector consistency
│   ├── structural_loss.py      # Differentiable structural loss (1 - SSIM)
│   ├── identity_regularization.py # Residual magnitude regularizer
│   └── total_loss.py           # Multi-term composite loss manager
├── metrics/
│   ├── reference_metrics.py    # PSNR, SSIM, MSE, MAE, optional LPIPS
│   ├── no_reference_metrics.py # Shannon entropy, RMS contrast
│   └── efficiency.py           # FLOPs, parameter counter, latency profiler
├── scripts/
│   ├── train.py                # Self-supervised training with checkpoints
│   ├── infer.py                # Inference script with diagnostic maps
│   ├── evaluate.py             # Evaluation across paired & unpaired sets
│   ├── benchmark.py            # Latency, FPS, parameter & FLOPs benchmark
│   └── run_ablation.py         # Automated ablation matrix generator
├── tests/
│   └── test_lumaguard.py       # Complete pytest test suite
└── docs/
    ├── architecture.md         # Mermaid architecture diagram & module details
    ├── mathematical_formulation.md # Full mathematical equations & tensor specs
    ├── experimental_protocol.md # Datasets, splits, evaluation guidelines
    └── limitations.md          # Scientific caveats and inverse problem trade-offs
```

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python -m pytest tests/ -v
```

### 3. Run Benchmark
```bash
python scripts/benchmark.py --height 400 --width 600 --device cpu
```

### 4. Run Smoke Test Training & Diagnostics
```bash
# Generate synthetic samples
python scripts/generate_synthetic_data.py

# Train 3 epochs on CPU
python scripts/train.py --config configs/smoke_test.yaml

# Run inference with all diagnostic map exports
python scripts/infer.py \
  --input data/synthetic_sample/val/low/synthetic_val_00.png \
  --checkpoint experiments/smoke_test/best_checkpoint.pth \
  --output outputs/smoke_inference \
  --diagnostics
```

---

## Citation & References
1. **SNR-Aware Low-Light Image Enhancement**, CVPR 2022.
2. **Deep Retinex Decomposition for Low-Light Enhancement (LOL)**, BMVC 2018.
3. **Exclusively Dark Image Dataset (ExDark)**, IEEE TIP 2020.
