# Losses

This package contains the objective functions used during training.

## Included module

- `paired_rgb_restoration.py`: paired-image loss formulation for low-light image restoration

## Purpose

The restoration model is trained with supervision from paired low-light and normal-light reference images. The loss implementation is designed to encourage pixel-level reconstruction quality while preserving color fidelity and structural detail.

## Notes

- Loss functions are selected and configured from the training script.
- For paired training, the reference images are the normal-light targets.
- For unpaired evaluation, no-reference metrics are used instead of paired supervision.
