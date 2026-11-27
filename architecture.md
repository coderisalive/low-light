# Architecture

## Overview

This project implements a paired low-light image enhancement pipeline built around a lightweight U-Net style neural network. The system learns a mapping from degraded low-light RGB inputs to their corresponding normal-light targets.

The architecture is centered on:

- a multiscale encoder-decoder network
- skip connections for spatial detail preservation
- residual learning for stable reconstruction
- RGB-space correction to preserve color fidelity

## Core components

### 1. Data pipeline

The dataset layer in `datasets/` prepares image pairs for training and evaluation. It discovers dataset-specific folders and normalizes image loading for:

- LOL-v1
- LOL-v2-Real
- LOL-v2-Synthetic
- ExDark (unpaired benchmark)

The code keeps paired datasets separate from unpaired evaluation logic so that reference-based and no-reference metrics are not mixed incorrectly.

### 2. Model

The main model lives in `models/paired_rgb_restoration_net.py` and follows a U-Net-style design with:

- encoder blocks that reduce spatial dimensions and increase feature depth
- decoder blocks that reconstruct high-resolution outputs
- skip connections that merge low-level and high-level features
- depthwise-separable residual blocks for efficiency
- a residual correction stage that improves RGB reconstruction quality

### 3. Loss function

The objective in `losses/paired_rgb_restoration.py` is designed for paired supervised enhancement. It encourages the model to approximate the target image under low-light degradations while preserving structure and vivid color distribution.

### 4. Training loop

The training script `scripts/train_rgb_restoration.py` controls:

- dataset loading
- balanced or proportional sampling
- epoch scheduling
- early stopping behavior
- checkpoint saving
- validation tracking

### 5. Evaluation and inference

The evaluation pipeline in `scripts/evaluate_rgb_restoration.py` measures model quality on official paired and unpaired test sets. Inference is handled separately in `scripts/infer.py`, which is intended for running the trained model on new image inputs.

## Data flow

```text
Low-light image
    ↓
Dataset loader
    ↓
Preprocessing / transforms
    ↓
Encoder network
    ↓
Decoder + skip connections
    ↓
Output restoration image
    ↓
Loss comparison with target
```

## Design principles

- keep the model lightweight enough for practical training
- support multiple low-light datasets through one common pipeline
- isolate official test sets from training data
- use paired metrics for paired data and no-reference metrics for unpaired data
- favor reproducible experiments via clear folder and checkpoint organization

## File structure reference

- `models/` — model architectures
- `datasets/` — dataset loading and preprocessing
- `losses/` — objective functions
- `scripts/` — training, evaluation, and inference CLI scripts
- `metrics/` — evaluation metrics
- `experiments/` — saved training runs and outputs
- `outputs/` — generated comparison results and summaries
