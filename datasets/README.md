# Datasets

This package contains the dataset loaders and preprocessing helpers used to train and evaluate the restoration model.

## Included modules

- `multi_dataset.py`: multi-dataset discovery and loading for LOL-v1, LOL-v2-Real, and LOL-v2-Synthetic
- `exdark_dataset.py`: dataset wrapper for the unpaired ExDark benchmark
- `paired_evaluation_dataset.py`: paired evaluation dataset utilities
- `transforms.py`: image transforms and augmentation helpers

## Supported datasets

- LOL-v1
- LOL-v2-Real
- LOL-v2-Synthetic
- ExDark

## Usage

The loaders are expected to be used by the training and evaluation scripts in [scripts/README.md](../scripts/README.md). They automatically search for common folder naming patterns and support both official dataset layouts and Hugging Face-style directory structures.

## Data conventions

- paired datasets use low-light and normal-light image pairs
- ExDark is treated as an unpaired benchmark and is evaluated with no-reference metrics
- evaluation on official test splits should remain isolated from training data
