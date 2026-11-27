# Metrics

This package provides evaluation metrics for low-light image enhancement.

## Included module

- `reference_metrics.py`: full-reference metrics for paired restoration benchmarks

## Metrics covered

- PSNR
- SSIM
- MSE
- MAE
- information entropy
- RMS contrast

## Usage

Evaluation scripts compute metrics from model outputs and ground-truth references. The metrics are used to compare restoration quality across low-light datasets and to create summaries in CSV output files.

## Notes

- Paired datasets should use reference-based metrics.
- Unpaired datasets such as ExDark should use no-reference metrics and visual inspection.
