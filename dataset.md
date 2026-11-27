# Dataset

## Purpose

This project trains and evaluates a low-light image enhancement model using paired and unpaired datasets. The goal is to restore normal-light color, contrast, and detail from degraded low-light inputs.

## Supported datasets

### 1. LOL-v1

- paired dataset
- training set: `our485`
- test set: `eval15`
- low-light and normal-light images are aligned by filename and folder

### 2. LOL-v2-Real

- paired dataset
- official folder layout under `LOL-v2-Real`
- supports common structures such as `Train/Input` and `Train/GT`
- also supports Hugging Face-style `*_hf_raw` directory arrangements

### 3. LOL-v2-Synthetic

- paired dataset
- similar structure to LOL-v2-Real
- used for larger-scale supervised training and benchmark evaluation

### 4. ExDark

- unpaired dataset
- no normal-light ground truth is available
- evaluated with no-reference metrics, not paired reconstruction losses

## Expected folder structure

```text
data/
├── lol_dataset/
│   ├── our485/
│   │   ├── low/
│   │   └── high/
│   └── eval15/
│       ├── low/
│       └── high/
├── LOL-v2-Real/
│   └── Real_captured/
│       └── _hf_raw/
├── LOL-v2-Synthetic/
│   └── Synthetic/
│       └── _hf_raw/
├── ExDark/
│   └── Groundtruth/
└── synthetic_sample/
```

## Important rules

- Paired datasets are used for supervised training and full-reference evaluation.
- ExDark is not used as a ground-truth pair source.
- Official test folders must stay isolated from training split selection.
- Data roots can be passed explicitly using script arguments, or auto-detected when the standard paths exist.

## Dataset code

The loader logic is primarily located in:

- `datasets/multi_dataset.py`
- `datasets/exdark_dataset.py`
- `datasets/paired_evaluation_dataset.py`

These modules detect dataset folder names and map them to the correct training or evaluation pipeline.

## Training vs evaluation usage

- training: use only training splits or data pools
- evaluation: use official testing folders or benchmark splits
- no-reference metrics are used for ExDark and similar unpaired benchmarks

## Notes

The repository supports both official dataset naming patterns and normalized folder variations, which makes it easier to run on local machines, Kaggle, or cloud workspaces without changing the model code.
