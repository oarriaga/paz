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
brightness). `scenes.build_mesh` normalizes any mesh to unit extent and the
camera is framed (distance 2.5) with an off-axis point light, so the object is
centered with margin and its faces are distinctly shaded. With no `--mesh`, a
solid per-face colored cube (six plain face colors) is used so the example runs
offline.

Camera poses are sampled with `paz.SO3.sample` (uniform rotations), and domain
randomization reuses the `paz.backend.image` pipeline:
`paz.image.randomize_rendered_image` blends the object over a background
(plain random color, or a random crop of a provided image), adds random
occlusions, applies blur and color jitter — mirroring the legacy
`RandomizeRenderedImage`. Pass `--backgrounds` to composite over real images.

The whole pipeline is JAX: `fill_polygon` and `apply_gaussian_blur` replace the
cv2 calls, so `randomize_rendered_image` is `jit`/`vmap`-able (the training
sequence jits a vmapped batch). At `128x128`, a jitted+vmapped batch runs about
**10-12x faster on GPU (RTX A4000) than cv2 per-image on CPU** (~0.7 ms vs
~7 ms for a batch of 32), and the views stay on-device with the renderer. On
CPU the JAX path is slower than cv2, so this targets GPU training.

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

## Evaluate

```bash
KERAS_BACKEND=jax python eval.py --weights experiments/aae.weights.h5
```

Renders a dense codebook and a held-out set of random orientations, retrieves
the nearest codebook pose for each (clean and domain-randomized), and reports
the geodesic angular error against the codebook-resolution oracle floor. It also
writes a `[clean | augmented | true axes / predicted axes]` montage to
`experiments/eval.png`. On the solid-color cube the augmented retrieval reaches
a **median of ~7 deg**, matching the clean result and the oracle floor.

A uniform-colored face is rotationally symmetric, so near face-on views cannot
resolve the in-plane rotation; those rare views form a high-error tail (mean
~20 deg) while typical two/three-face views are recovered exactly. Breaking each
face's symmetry (a small per-face accent) would remove the tail.

## Notes

- `paz.models.AutoEncoder((128,128,3), latent_dimension)` builds the model;
  `paz.models.extract_encoder(model)` returns the encoder for the codebook.
- Background compositing uses the renderer's constant white background as the
  foreground mask; for a textured object provide `--backgrounds`.
