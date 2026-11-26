# Changelog

All notable changes to PAZ are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 1.0.0 (unreleased)

PAZ 1.0 is a rewrite of the library, developed on the `paz-jax` branch over
roughly 1500 commits. It is **not backward compatible** with the 0.2 series.

### Highlights

- **JAX-first rewrite on Keras 3.** Backend functions are pure, operate on
  JAX arrays and compose with `jax.jit`, `jax.vmap` and `jax.grad`.
- **Differentiable ray tracer** in `paz.graphics`: scene compilation, cameras,
  shapes and meshes, shadows, and Phong and Cook-Torrance shading.
- **Probabilistic programming** in `paz.inference`: priors, latents and
  observables, probabilistic graphical models, Metropolis-Hastings with an
  adaptive covariance tuner, and native distributions and bijectors
  (`paz.distributions`, `paz.bijectors`).
- **Optimization** in `paz.optimization`: minimization with line search,
  golden-section search, restarts, stopping criteria and callbacks.
- **Lie groups**: `paz.SO3`, `paz.SE3`, `paz.SE2` and `paz.quaternion`.
- **Foundation-model ports** in `paz.models.foundation`: DINOv2, DINOv3,
  Depth Anything 3, SAM 2 (image and video), Gemma 4, Whisper, Florence-2,
  FLOWER, SONIC and GEAR-WBC.
- **Detection and feature models**: SSD300/SSD512, EfficientDet, RF-DETR and
  PF-RPN in `paz.models.detection`; XFeat and LightGlue in
  `paz.models.feature`.
- **Shared transformer blocks** in `paz.models.transformers`.
- **Ready-to-use applications** in `paz.applications`: detectors, segmenters,
  trackers, depth, keypoint and pose estimators, feature matchers,
  transcribers and generators.
- **Utilities**: remote function calls (`paz.remote`), on-disk caching
  (`paz.cache`), audio and video backends, point clouds, planes and meshes.

### Changed

- The package is published on PyPI as `pypaz`; the import name stays `paz`.
- Python >= 3.10 is required.
- The Keras backend defaults to JAX; an explicit `KERAS_BACKEND` is respected.
- OpenCV is pinned to `opencv-python>=4.8,<5`.
- TensorFlow and TensorFlow Probability are no longer dependencies.
- Tests are co-located with their modules as `*_test.py`.
- `paz.backend.groups` was renamed to `paz.backend.lie`.
- `paz.backend.image` is a single module instead of a package with OpenCV
  and TensorFlow variants.
- SSD models moved to `paz.models.detection.single_shot_detector`.
- The documentation is built from Markdown guides by `docs/build.py`.

### Removed

- `paz.processors` and `paz.pipelines`.
- `SequentialProcessor`, `Processor`, `Loader`, `ProcessingSequence` and
  `GeneratingSequence` from `paz.abstract`.
- The `paz.evaluation` package (`evaluateMAP`, `EvaluateADD`, `EvaluateADI`).
- EfficientPose (`paz.models.pose_estimation.efficientpose` and its example).
- The old documentation generator: `docs/templates`, `docs/structure.py` and
  `docs/autogenerate_documentation.py`.
- Legacy examples: `discovery_of_latent_keypoints`, `face_classification`,
  `fine-tuning_object_detection`, `haar_cascade_detectors`,
  `images_synthesis`, `mask_rcnn`, `tutorials` and
  `visual_voice_activity_detection`.

### Migrating from 0.2

1.0 is not backward compatible: code written for 0.2 must be ported. Install
with `pip install pypaz` and keep `import paz`.

| 0.2 | 1.0 |
| --- | --- |
| `paz.backend.image.load_image` | `paz.image.load` |
| `paz.backend.image.show_image` | `paz.image.show` |
| `paz.backend.image.resize_image` | `paz.image.resize` |
| `paz.backend.image.write_image` | `paz.image.write` |
| `paz.backend.image.draw_circle` | `paz.draw.circle` |
| `paz.backend.image.draw_rectangle` | `paz.draw.rectangle` |
| `paz.backend.groups` | `paz.backend.lie` (`paz.SO3`, `paz.SE3`) |
| `paz.evaluation.evaluateMAP` | `paz.evaluation.compute_mAP` |
| `paz.pipelines.*` | factories in `paz.applications` |
| `SequentialProcessor` | plain functions, or `paz.Sequential` |

Processors have no one-to-one replacement: preprocessing and postprocessing
are now plain functions in `paz.image`, `paz.boxes`, `paz.detection`,
`paz.draw` and the other backend modules.
