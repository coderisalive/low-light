# Configs

This directory is intended for reusable training and inference configuration files.

Use it to store:

- dataset path presets
- experiment configuration templates
- hyperparameter groups for different runs
- reproducibility notes for model training

## Typical usage

Keep config values aligned with the command-line arguments exposed by the training and evaluation scripts under [scripts/README.md](../scripts/README.md).

## Example

A config may define:

- dataset roots
- image crop size
- batch size
- learning rate
- optimizer settings
- checkpoint output directory

When possible, prefer versioning config files with the corresponding experiment name in the run folder under [experiments/README.md](../experiments/README.md).
