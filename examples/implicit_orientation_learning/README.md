# Implicit orientation learning

Augmented-autoencoder (AAE) baseline for implicit 3D orientation estimation.
A convolutional autoencoder is trained to reconstruct clean rendered views of
an object from heavily augmented versions of the same view. The encoder then
maps a crop to a latent vector; matching that latent against a codebook of
rendered views at known poses (cosine similarity) yields the orientation.

This is a faithful port of the legacy `master` example to JAX / Keras-3.

## Rendering

Synthetic views are produced with `paz.graphics` (the built-in renderer) — no
external rasterizer. Views are rendered once and cached to
`experiments/views.npz`, since ray tracing is the slow step; training then
augments the cached views each epoch (random background, occlusions,
brightness). `scenes.build_mesh` normalizes any mesh to unit extent, so the
default camera distance frames any object. With no `--mesh`, a colored cube is
used so the example runs offline.

Camera poses are sampled with `paz.SO3.sample` (uniform rotations), and domain
randomization reuses the `paz.backend.image` pipeline:
`paz.image.randomize_rendered_image` blends the object over a background
(plain random color, or a random crop of a provided image), adds random
occlusions, applies blur and color jitter — mirroring the legacy
`RandomizeRenderedImage`. Pass `--backgrounds` to composite over real images.

The whole pipeline is JAX: `fill_polygon` and `gaussian_blur` replace the cv2
calls, so `randomize_rendered_image` is `jit`/`vmap`-able (the training
sequence jits a vmapped batch). On CPU this is slower than cv2 per image; the
JAX path pays off on GPU and keeps data generation differentiable and on the
same device as the renderer.

## Train

```bash
KERAS_BACKEND=jax python train.py --root experiments
# or with a textured object and real backgrounds:
KERAS_BACKEND=jax python train.py --mesh model.obj --backgrounds voc/ \
    --num_views 5000 --epochs 300
```

Writes `experiments/aae.weights.h5`. Input is `128x128`.

## Demo

```bash
KERAS_BACKEND=jax python demo.py --weights experiments/aae.weights.h5
```

Builds a 10x10 pose codebook from the same mesh, encodes each camera frame,
and shows the nearest codebook view (the implicit orientation). No weights are
hosted — train first to produce the checkpoint.

## Notes

- `paz.models.AutoEncoder((128,128,3), latent_dimension)` builds the model;
  `paz.models.extract_encoder(model)` returns the encoder for the codebook.
- Background compositing uses the renderer's constant white background as the
  foreground mask; for a textured object provide `--backgrounds`.
