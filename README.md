# PAZ

PAZ is a perception library written in JAX on Keras 3: differentiable rendering, probabilistic programming, computer vision, foundation models, and robotics and control.

[![Tests](https://github.com/oarriaga/paz/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/oarriaga/paz/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/pypaz.svg)](https://pypi.org/project/pypaz/)
[![Documentation](https://img.shields.io/badge/docs-oarriaga.github.io%2Fpaz-blue.svg)](https://oarriaga.github.io/paz)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<!-- hero image: scripts/readme/hero.py -->

## Areas

| Area | What it offers | Subpackage | Example |
|---|---|---|---|
| Differentiable rendering | Ray tracing of shapes and meshes with Phong and Cook-Torrance shading, cameras, lights and patterns; renders are JAX functions that can be passed to `jax.grad`, `jax.jit` and `jax.vmap` | `paz.graphics` | [differentiable_shape_rendering](examples/differentiable_shape_rendering) |
| Probabilistic programming | Priors, observables and probabilistic graphical models, Metropolis-Hastings sampling with adaptive tuners, bijector fitting, serialization | `paz.inference` | [probabilistic_programming](examples/probabilistic_programming) |
| Computer vision | Image, box, keypoint and geometry functions; detection, classification, keypoint, pose and segmentation models; application factories that return a ready-to-call function | `paz.backend`, `paz.models`, `paz.applications` | [rf_detr](examples/rf_detr) |
| Foundation models | Depth Anything 3, DINOv2, DINOv3, Florence-2, Gemma 4, SAM 2 and Whisper | `paz.models.foundation` | [sam2](examples/sam2) |
| Robotics and control | FLOWER vision-language-action model, SONIC and GEAR-WBC humanoid controllers, PPO training for legged locomotion | `paz.models.foundation` (`flower`, `sonic`, `gear_wbc`) | [rl_gym](examples/rl_gym) |

## Installation

```bash
pip install pypaz
```

PAZ requires Python >= 3.10. `import paz` sets the Keras backend to JAX unless the `KERAS_BACKEND` environment variable is already set.

Optional extras, installed with `pip install "pypaz[name]"`:

| Extra | Adds | Used by |
|---|---|---|
| `weights` | `safetensors`, `huggingface_hub`, `keras_hub` | converting upstream checkpoints |
| `datasets` | `gdown`, `kaggle`, `cityscapesscripts` | dataset download helpers |
| `sonic` | `mujoco`, `onnx` | [examples/sonic](examples/sonic) |
| `gear_wbc` | `mujoco`, `onnx`, `pygame` | [examples/gear_wbc](examples/gear_wbc) |
| `learning_to_walk` | `mujoco`, `pygame`, `pyyaml`, `torch` | [examples/learning_to_walk](examples/learning_to_walk) |

## Quickstart

Pretrained weights are downloaded on first use and cached under `~/.keras/paz`.

### Object detection

```python
import paz
from paz.applications import DetectRFDETRNano

detect = DetectRFDETRNano(0.5)
image = paz.image.load("image.jpg")
(boxes, labels, scores), drawn = detect(image)
names = ["0"] + paz.datasets.labels("COCO_EFFICIENTDET")
for label, score in zip(labels, scores):
    print(names[label], float(score))
paz.image.write("detections.png", drawn)
```

### Differentiable rendering

The gradient of a pixel loss with respect to a material color:

```python
import jax
import jax.numpy as jp
import paz
from paz.graphics import Material, PointLight, Scene, Sphere

shape = (32, 32)
intrinsics = paz.graphics.camera.compute_intrinsics(jp.pi / 3.0, *shape)
eye, up = jp.array([0.0, 0.0, 5.0]), jp.array([0.0, 1.0, 0.0])
pose = paz.SE3.view_transform(eye, jp.zeros(3), up)
lights = [PointLight(jp.ones(3), jp.array([0.0, 5.0, 5.0]))]

def compute_loss(color):
    scene = Scene([Sphere(jp.eye(4), Material(color=color))])
    args = shape, intrinsics, pose, scene, None, lights, (1, 1), 1024
    image, depth = paz.graphics.render(*args)
    return jp.mean((image - 0.5) ** 2)

gradient = jax.jit(jax.grad(compute_loss))(jp.array([0.8, 0.2, 0.1]))
print(gradient)
```

### Probabilistic programming

A normal prior over a mean, 20 observations, and four Metropolis-Hastings chains:

```python
import jax
import jax.numpy as jp
import paz

def likelihood(mu):
    return paz.distributions.Normal(jp.full(20, mu), 1.0)

mu = paz.Prior(paz.distributions.Normal(0.0, 2.0), name="mu")
x = paz.Observable(likelihood, name="x")(mu)
model = paz.PGM([mu], [x], "normal_normal")
key_data, key_infer = jax.random.split(jax.random.PRNGKey(0))
observations = 3.0 + jax.random.normal(key_data, (20,))
model.configure(num_chains=4, warmup=0.2, sigma=0.2)
posterior = model.infer(key_infer, observations, num_samples=2000)
print(posterior.samples.mu.mean(), posterior.samples.mu.std())
```

### Optimization

`paz.optimization.minimize` runs any optax optimizer and returns the status, the parameters with the lowest loss, and the loss history:

```python
import jax.numpy as jp
import optax
import paz

def compute_loss(parameters):
    return jp.sum((parameters - jp.array([1.0, -3.0])) ** 2)

parameters = jp.array([8.0, -2.0])
optimizer = optax.adam(0.1)
minimize = paz.optimization.minimize
status, fitted, history = minimize(parameters, compute_loss, optimizer, 500)
print(fitted, history.losses[-1])
```

### Foundation models

Depth from one or more views with Depth Anything 3:

```python
import paz
from paz.applications import EstimateDepthAnything3Small

estimate = EstimateDepthAnything3Small()
image = paz.image.load("image.jpg")
outputs = estimate([image])
depth, confidence, extrinsics, intrinsics, rays, ray_confidence = outputs
print(depth.shape, float(depth.min()), float(depth.max()))
```

## Models

Application factories are listed in [paz/applications/\_\_init\_\_.py](paz/applications/__init__.py).

| Model | Task | Module | Example |
|---|---|---|---|
| SSD300, SSD512 | Object detection | [paz/models/detection/single_shot_detector](paz/models/detection/single_shot_detector) | [object_detection](examples/object_detection) |
| EfficientDet D0-D7 | Object detection | [paz/models/detection/efficientdet](paz/models/detection/efficientdet) | [efficientdet](examples/efficientdet) |
| RF-DETR Nano, Small, Medium, Base, Large | Object detection | [paz/models/detection/rf_detr](paz/models/detection/rf_detr) | [rf_detr](examples/rf_detr) |
| PF-RPN | Class-agnostic object proposals | [paz/models/detection/pf_rpn](paz/models/detection/pf_rpn) | [pf_rpn](examples/pf_rpn) |
| Haar cascades | Face and eye detection | [paz/models/detection/haar_cascade.py](paz/models/detection/haar_cascade.py) | [face_detection](examples/face_detection) |
| MiniXception | Emotion and gender classification | [paz/models/classification/xception.py](paz/models/classification/xception.py) | [emotion_classifier](examples/emotion_classifier) |
| Prototypical networks | Few-shot classification | [paz/models/classification/protonet.py](paz/models/classification/protonet.py) | [prototypical_networks](examples/prototypical_networks) |
| KeypointNet2D | Face keypoints | [paz/models/keypoint/keypointnet.py](paz/models/keypoint/keypointnet.py) | [keypoint_estimation](examples/keypoint_estimation) |
| Gaussian mixture keypoints | Keypoints with uncertainty | [paz/models/keypoint/gaussian_mixture.py](paz/models/keypoint/gaussian_mixture.py) | [probabilistic_keypoint_estimation](examples/probabilistic_keypoint_estimation) |
| DetNet, IKNet | Hand keypoints and joint angles | [paz/models/keypoint/detnet.py](paz/models/keypoint/detnet.py), [iknet.py](paz/models/keypoint/iknet.py) | [hand_pose_estimation](examples/hand_pose_estimation) |
| HigherHRNet | 2D human pose | [paz/models/pose_estimation/higher_hrnet.py](paz/models/pose_estimation/higher_hrnet.py) | [human_pose_estimation_2D](examples/human_pose_estimation_2D) |
| SimpleBaselines | 3D human pose from 2D keypoints | [paz/models/keypoint/simplebaselines.py](paz/models/keypoint/simplebaselines.py) | [human_pose_estimation_3D](examples/human_pose_estimation_3D) |
| UNet | Semantic segmentation | [paz/models/segmentation/unet.py](paz/models/segmentation/unet.py) | [semantic_segmentation](examples/semantic_segmentation) |
| Spatial transformer network | Attention | [paz/models/attention/stn.py](paz/models/attention/stn.py) | [spatial_transformer_networks](examples/spatial_transformer_networks) |
| Autoencoder | Representation learning | [paz/models/autoencoder/autoencoder.py](paz/models/autoencoder/autoencoder.py) | [implicit_orientation_learning](examples/implicit_orientation_learning) |
| XFeat, LightGlue | Local features and matching | [paz/models/feature/xfeat](paz/models/feature/xfeat), [lightglue](paz/models/feature/lightglue) | |
| PCA | Decomposition | [paz/models/decomposition/pca.py](paz/models/decomposition/pca.py) | [eigenfaces](examples/eigenfaces) |
| K-means, DBSCAN | Clustering | [paz/models/clustering](paz/models/clustering) | |
| Depth Anything 3 | Depth, rays and cameras from one or more views | [paz/models/foundation/depth_anything3](paz/models/foundation/depth_anything3) | [depth_anything3](examples/depth_anything3) |
| DINOv2 | Image features | [paz/models/foundation/dinov2](paz/models/foundation/dinov2) | |
| DINOv3 (weights not hosted yet) | Image features | [paz/models/foundation/dinov3](paz/models/foundation/dinov3) | |
| Florence-2 | Multimodal encoder used by FLOWER | [paz/models/foundation/florence2](paz/models/foundation/florence2) | [flower](examples/flower) |
| FLOWER | Vision-language-action policy | [paz/models/foundation/flower](paz/models/foundation/flower) | [flower](examples/flower) |
| Gemma 4 | Text generation and image description | [paz/models/foundation/gemma4](paz/models/foundation/gemma4) | [gemma4](examples/gemma4) |
| SAM 2 | Image segmentation and video tracking | [paz/models/foundation/sam2](paz/models/foundation/sam2) | [sam2](examples/sam2) |
| Whisper (English variants) | Speech to text | [paz/models/foundation/whisper](paz/models/foundation/whisper) | [speech_to_text](examples/speech_to_text) |
| SONIC | Humanoid whole-body control | [paz/models/foundation/sonic](paz/models/foundation/sonic) | [sonic](examples/sonic) |
| GEAR-WBC | Humanoid lower-body control | [paz/models/foundation/gear_wbc](paz/models/foundation/gear_wbc) | [gear_wbc](examples/gear_wbc) |

## Examples

**Differentiable rendering:**
[differentiable_rendering](examples/differentiable_rendering),
[differentiable_shape_rendering](examples/differentiable_shape_rendering),
[differentiable_mesh_rendering](examples/differentiable_mesh_rendering),
[multi_view_mesh_optimization](examples/multi_view_mesh_optimization)

**Probabilistic programming:**
[probabilistic_programming](examples/probabilistic_programming),
[probabilistic_keypoint_estimation](examples/probabilistic_keypoint_estimation)

**Computer vision:**
[object_detection](examples/object_detection),
[efficientdet](examples/efficientdet),
[rf_detr](examples/rf_detr),
[pf_rpn](examples/pf_rpn),
[face_detection](examples/face_detection),
[hand_detection](examples/hand_detection),
[emotion_classifier](examples/emotion_classifier),
[imdb_classifier](examples/imdb_classifier),
[keypoint_estimation](examples/keypoint_estimation),
[hand_pose_estimation](examples/hand_pose_estimation),
[head_pose_estimation_6D](examples/head_pose_estimation_6D),
[human_pose_estimation_2D](examples/human_pose_estimation_2D),
[human_pose_estimation_3D](examples/human_pose_estimation_3D),
[pix2pose](examples/pix2pose),
[implicit_orientation_learning](examples/implicit_orientation_learning),
[semantic_segmentation](examples/semantic_segmentation),
[unsupervised_part_segmentation](examples/unsupervised_part_segmentation),
[plane_detection](examples/plane_detection),
[structure_from_motion](examples/structure_from_motion),
[eigenfaces](examples/eigenfaces)

**Learning methods:**
[prototypical_networks](examples/prototypical_networks),
[model_agnostic_meta_learning](examples/model_agnostic_meta_learning),
[spatial_transformer_networks](examples/spatial_transformer_networks),
[action_scores](examples/action_scores)

**Foundation models:**
[depth_anything3](examples/depth_anything3),
[sam2](examples/sam2),
[gemma4](examples/gemma4),
[speech_to_text](examples/speech_to_text),
[remote](examples/remote)

**Robotics and control:**
[flower](examples/flower),
[sonic](examples/sonic),
[gear_wbc](examples/gear_wbc),
[learning_to_walk](examples/learning_to_walk),
[rl_gym](examples/rl_gym)

## Documentation

Full documentation is at [https://oarriaga.github.io/paz](https://oarriaga.github.io/paz).

## Tests

```bash
pytest paz/ tests/
```

Tests that download pretrained weights are opt-in: they are skipped unless their environment variable is set, for example `GEMMA4_WEIGHTS_TEST=1`.

## Contributing

See `CONTRIBUTING.md`.

## Migrating from 0.2

PAZ 1.0 is a rewrite on JAX and Keras 3 and is not backward compatible with the TensorFlow-based 0.2 releases. See `CHANGELOG.md` for what changed.

## Citation
If you use PAZ please consider citating it. You can also find our paper here [https://arxiv.org/abs/2010.14541](https://arxiv.org/abs/2010.14541).

```BibTeX
@misc{arriaga2020perception,
      title={Perception for Autonomous Systems (PAZ)},
      author={Octavio Arriaga and Matias Valdenegro-Toro and Mohandass Muthuraja and Sushma Devaramani and Frank Kirchner},
      year={2020},
      eprint={2010.14541},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Funding
PAZ is currently developed in the [Robotics Group](https://robotik.dfki-bremen.de/de/ueber-uns/universitaet-bremen-arbeitsgruppe-robotik.html) of the [University of Bremen](https://www.uni-bremen.de/), together with the [Robotics Innovation Center](https://robotik.dfki-bremen.de/en/startpage.html) of the **German Research Center for Artificial Intelligence** (DFKI) in **Bremen**.
PAZ has been funded by the German Federal Ministry for Economic Affairs and Energy and the [German Aerospace Center](https://www.dlr.de/DE/Home/home_node.html) (DLR).
PAZ been used and/or developed in the projects [TransFIT](https://robotik.dfki-bremen.de/en/research/projects/transfit.html) and [KiMMI-SF](https://robotik.dfki-bremen.de/en/research/projects/kimmi-sf/).

<p align="center">
<img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/funding_partners.png" width="1200">
</p>
