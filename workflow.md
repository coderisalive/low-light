# Workflow

## End-to-end project workflow

### 1. Prepare datasets

Place the required dataset folders under `data/` or provide the dataset roots directly to the scripts.

Typical setup:

- `data/lol_dataset`
- `data/LOL-v2-Real`
- `data/LOL-v2-Synthetic`
- `data/ExDark`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Use the training script to run supervised low-light enhancement training:

```bash
python -u scripts/train_rgb_restoration.py \
  --lolv1_root data/lol_dataset \
  --lolv2_real_root data/LOL-v2-Real \
  --lolv2_synthetic_root data/LOL-v2-Synthetic \
  --sampling_mode balanced \
  --epochs 25 \
  --batch_size 8 \
  --crop_size 192 \
  --lr 0.0002
```

This step creates checkpoints and validation artifacts in the `experiments/` directory.

### 4. Evaluate the checkpoint

Once training is complete, benchmark the best checkpoint against official test sets:

```bash
python -u scripts/evaluate_rgb_restoration.py \
  --checkpoint experiments/<run_name>/best_checkpoint.pth \
  --lolv1_root data/lol_v1 \
  --lolv2_real_root data/LOL-v2-Real \
  --lolv2_synthetic_root data/LOL-v2-Synthetic \
  --exdark_root data/ExDark \
  --output_dir experiments/<run_name>_results
```

### 5. Run inference on new images

```bash
python scripts/infer.py --checkpoint <path_to_model> --input_dir <folder_of_low_light_images>
```

### 6. Inspect outputs

Generated outputs are usually stored under:

- `experiments/`
- `outputs/`
- `samples/` folders inside experiment runs

## Recommended operation sequence

1. verify dataset folders and image counts
2. train on paired datasets
3. monitor validation metrics and checkpoint quality
4. run benchmark evaluation on official test splits
5. use inference for sample or production images
6. compare outputs visually and numerically

## Common project stages

```text
Data source
  ↓
Preprocessing
  ↓
Training
  ↓
Validation
  ↓
Checkpoint selection
  ↓
Benchmark evaluation
  ↓
Inference / deployment use
```

## Notes

- Keep training and test splits separate.
- For ExDark, use no-reference metrics instead of paired loss evaluation.
- Save experiment metadata so each run remains reproducible.
