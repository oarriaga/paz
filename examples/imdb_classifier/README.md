# IMDB face attribute classifier

Two-class face attribute classification with mini_XCEPTION, trained on the
IMDB face dataset (`man` / `woman` labels). This mirrors the emotion
(`emotion_classifier`) example: a small Xception is run on grayscale face
crops, then composed with the Haar-cascade face detector for per-face
classification.

The public applications are `ClassifyMiniXceptionIMDB` (single face crop) and
`DetectMiniXceptionIMDB` (detect faces, then classify each).

## Architecture

`paz.models.build_mini_xception_imdb` reproduces the original
`oarriaga/face_classification` mini_XCEPTION (8-kernel stem, strided fourth
module, `64x64x1` input) so the released weights load by topological order.
A leading `Rescaling` maps the `[0, 1]` classifier input to the `[-1, 1]`
training range, so callers feed plain `[0, 1]` grayscale.

## Weights

The pretrained weights are the gender model from `oarriaga/face_classification`
(`gender_mini_XCEPTION`), converted to Keras-3 as
`imdb_mini_XCEPTION_paz_jax.weights.h5` (bit-identical outputs). Once that file
is hosted on `oarriaga/altamira-data`, `paz.models.MiniXceptionIMDB` loads it
directly. Until then, point the demo at a local copy.

## Data (for retraining)

Training is loader-free: provide grayscale `64x64x1` face crops (`uint8`) and
one-hot labels as NumPy arrays under `--data`:

```
data/train_images.npy        # (N, 64, 64, 1)
data/train_labels.npy        # (N, 2) one-hot
data/validation_images.npy
data/validation_labels.npy
```

## Train

```bash
KERAS_BACKEND=jax python train.py --data data --root experiments
```

This writes `experiments/imdb_mini_XCEPTION_paz_jax.weights.h5`.

## Demo

```bash
KERAS_BACKEND=jax python demo.py \
    --weights experiments/imdb_mini_XCEPTION_paz_jax.weights.h5
```

The demo loads a local checkpoint through the shared
`paz.applications.ClassifyMiniXception` / `DetectMiniXception` helpers, so it
runs without any hosted weights.
