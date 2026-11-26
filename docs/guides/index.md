# PAZ

PAZ (Perception for Autonomous Systems) is a perception library written in
[JAX](https://github.com/jax-ml/jax).

## Installation

```bash
pip install pypaz
```

## Areas

- **Differentiable rendering**: `paz.graphics` builds scenes, cameras and
  shapes and renders them with a ray tracer. See
  [renderer](api/graphics/renderer.md), [camera](api/graphics/camera.md),
  [scene](api/graphics/scene.md) and [shapes](api/graphics/shapes.md).
- **Probabilistic programming**: `paz.inference` holds probabilistic
  graphical models, densities and Metropolis-Hastings sampling. See
  [infer](api/inference/infer.md) and
  [metropolis_hastings](api/inference/metropolis_hastings.md).
- **Computer vision**: `paz.backend` holds functions for images, boxes,
  detection, drawing and Lie groups; `paz.applications` holds ready-to-use
  detectors and segmenters; `paz.losses` and `paz.optimization` are used to
  train and fit models. See [boxes](api/backend/boxes.md),
  [image](api/backend/image.md) and
  [detectors](api/applications/detectors.md).
- **Foundation models**: `paz.models.foundation` holds ports of pretrained
  models such as DINOv2, Depth Anything 3, SAM2, Whisper and Gemma 4.
- **Robotics and control**: `paz.models.foundation` also holds the FLOWER
  vision-language-action model and the GEAR-WBC and SONIC whole-body
  controllers.

## API reference

The pages under `api/` are generated from the docstrings in the source code
by `docs/build.py`. [Writing docstrings](docstrings.md) explains the
convention those docstrings follow.
