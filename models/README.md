# Models

This package contains the neural network implementations used by the image enhancement pipeline.

## Included module

- `paired_rgb_restoration_net.py`: Paired RGB restoration U-Net model

## Architecture overview

The model uses:

- an encoder-decoder structure
- skip connections
- multiscale feature extraction
- depthwise-separable residual blocks
- residual correction in RGB/logit space

## Notes

The main model is intended for paired supervised restoration of low-light RGB images and is the default network used by the training and evaluation scripts.
