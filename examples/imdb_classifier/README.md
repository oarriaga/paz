# IMDB face attribute classifier

Two-class face attribute classification with MiniXception, trained on the
IMDB face dataset (`man` / `woman` labels). This mirrors the emotion
(`emotion_classifier`) example: a `MiniXception` is trained on grayscale
`48x48` face crops, then composed with the Haar-cascade face detector for
per-face classification.

The model architecture is shared with the emotion path; only the head size
(`2` classes) and the dataset differ. The public applications are
`ClassifyMiniXceptionIMDB` (single face crop) and `DetectMiniXceptionIMDB`
(detect faces, then classify each).

## Weights

No pretrained weights are hosted yet. `paz.models.MiniXceptionIMDB` points at
the intended release path, but the file must be produced by training first and
then uploaded. Until then, train locally and run the demo against the produced
checkpoint.

## Data

Training is loader-free: provide grayscale `48x48x1` face crops and one-hot
labels as NumPy arrays under `--data`:

```
data/train_images.npy        # (N, 48, 48, 1)
data/train_labels.npy        # (N, 2) one-hot
data/validation_images.npy
data/validation_labels.npy
```

## Train

```bash
KERAS_BACKEND=jax python train.py --data data --root experiments
```

This writes `experiments/imdb_mini_XCEPTION_paz_jax.weights.h5`, ready to host
on `oarriaga/altamira-data` (which then enables `MiniXceptionIMDB`).

## Demo

```bash
KERAS_BACKEND=jax python demo.py \
    --weights experiments/imdb_mini_XCEPTION_paz_jax.weights.h5
```

The demo loads the locally trained checkpoint through the shared
`paz.applications.ClassifyMiniXception` / `DetectMiniXception` helpers, so it
runs without any hosted weights.
