# Scripts

This folder contains the command-line entry points for the project.

## Included scripts

- `train_rgb_restoration.py`: train the low-light enhancement model
- `evaluate_rgb_restoration.py`: benchmark a checkpoint across datasets
- `infer.py`: run inference on a single image or directory of images
- `download_lolv2.py`: download LOL-v2 and ExDark data into the local project structure

## Typical workflow

1. Prepare datasets in [data/README.md](../data/README.md)
2. Train the model with `train_rgb_restoration.py`
3. Evaluate with `evaluate_rgb_restoration.py`
4. Run inference with `infer.py`
5. Inspect outputs under [outputs/README.md](../outputs/README.md)

## Notes

The script arguments match the dataset locations and training configuration used throughout the project. Refer to the top-level README for full commands and examples.
