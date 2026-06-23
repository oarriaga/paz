# paz porting roadmap (legacy `master` → JAX/Keras-3)

Living document. The new `paz` is a JAX + Keras-3 perception library spanning
domain-specific perception (past), foundation models (present), and
probabilistic graphics (future). This file tracks which **domain-specific**
capabilities from the legacy `master` branch we have recovered and what is
still missing. A parallel effort ports foundation models; see the partition at
the bottom.

## Conventions for every port
- Clean functional style: models are functions returning a Keras `Model`;
  applications are functions with inner `preprocess`/`postprocess`/`call`
  closures. No `Processor`/`Pipeline`/`SequentialProcessor` classes.
- Follow `CLAUDE.md` (80 cols, no type hints, no docstrings on small helpers,
  `build_*`/`compute_*`, `box2D`/`box3D`, namedtuples for structured data).
- Weights: ported `*_paz_jax.weights.h5` hosted on
  `github.com/oarriaga/altamira-data/releases`; the maintainer uploads, then
  `WEIGHT_PATH` is wired. Verify by downloading fresh (sha256 + load + run).
- Verify on CPU (`CUDA_VISIBLE_DEVICES=""`); keep edits out of
  `paz/models/foundation/**`. Demos are image-verified; webcam paths smoke-only.

## Done (inference)
| Capability | App | Weights | Demo | Verified |
|---|---|---|---|---|
| SSD300-VOC / SSD512-COCO | ✅ | existing | object_detection | ✅ (fixed SSD512 priors) |
| Haar cascade | ✅ | — | face_detection | ✅ |
| MiniXception emotion (FER) | ✅ | v0.6 | emotion_classifier | ✅ |
| Face/Head KeypointNet2D32 | ✅ | v0.7 | head_pose / keypoint | ✅ |
| Minimal Hand (DetNet+IKNet) | ✅ | v0.14 | — | ✅ end-to-end |
| **Hand detection (SSD512 OIV6Hand)** | ✅ | v0.1 | hand_detection | ✅ boxes on hands |
| **SSD512-YCBVideo** | ✅ | v0.1 | — | ✅ YCB objects (DOPE img) |
| **SSD300-FAT** | ✅ | v0.2 | — | ✅ YCB/FAT objects |
| **Probabilistic keypoints (GMM)** | ✅ | **not hosted** | probabilistic_keypoint_estimation | model+math+loss+train verified |
| **DetectMinimalHand** (detect+pose) | ✅ | v0.1+v0.14 | hand_detection | ✅ |
| HigherHRNet 2D | ✅ | v0.10 | human_pose_estimation_2D | ✅ bit-exact + multi-person |
| SimpleBaselines 3D | ✅ | v0.17 | — | ✅ bit-exact |
| **Human pose 3D + 6D** (`EstimateHumanPose`) | ✅ | — | human_pose_estimation_3D | ✅ 6D opt + viz |
| EfficientDet D0–D7 (COCO) | ✅ | v0.16 | efficientdet | ✅ |
| EfficientPose (model) | model only | **not hosted** | — | architecture only |
| STN, ProtoNet, Xception, PCA, kNN/DBSCAN, MAML, eigenfaces | ✅ | n/a | various | ✅ |

## Done (training scripts)
| Script | Verified | Data dependency |
|---|---|---|
| SSD (`object_detection/train.py`) | pre-existing | VOC |
| **EfficientPose** (`efficientpose/train.py`) | ✅ `--smoke` (loss≈5.7) | LINEMOD + `.ply` |
| **Action scores** (`action_scores/train_classifier.py`) | ✅ synthetic | FER |
| **Hand detection** (`hand_detection/train.py`) | wired (reuses SSD pipeline) | Open Images V6 |
| **MiniXception FER** (`emotion_classifier/train.py`) | ✅ synthetic | FER |

## Still missing / not yet ported

### Tier 1 — high value
- **EfficientPose inference app** (`EstimateEfficientPose`): the model + training
  are ported, but there is **no inference application** (decode 6D pose, draw
  cube) and **no pretrained weights are hosted** (404 everywhere). Needs either
  a trained checkpoint (run `efficientpose/train.py` on LINEMOD) or an uploaded
  weights file. Also pending: 6D-aware geometric augmentation + ADD eval callback.
- **Pix2Pose** (`examples/pix2pose`): 6D object pose via pixel-wise 3D coords +
  GAN; includes the **DOPE** model. Multiple train + demo scripts. Not started.
  Bridges toward the future differentiable-rendering direction.

### Tier 2 — instance segmentation / detection variants
- **Mask R-CNN** (`examples/mask_rcnn`): RPN, ROIAlign, proposal/detection/mask
  heads, train + demos. Heaviest port. Not started.
- **EfficientDet-VOC** (`EFFICIENTDETD0VOC`) and **EfficientDet train.py /
  evaluate_mAP** — only COCO inference is done.
- **Extra SSD heads**: done. `SSD512-YCBVideo` verified end-to-end (v0.1
  weights) and `SSD300-FAT` verified end-to-end (`SSD300-FAT-FAT_weights.hdf5`
  at v0.2). Both detect YCB objects on the DOPE image.
- **Semantic segmentation (UNet)**: model is ported, but there is **no
  application, example, or train.py** (master had `train.py` + cityscapes + demo).

### Other domain capabilities not ported
- **implicit_orientation_learning** — augmented autoencoder for object
  orientation (train + demo).
- **visual_voice_activity_detection (VVAD)** — `cnn2Plus1` video model + demos.
- **discovery_of_latent_keypoints** — self-supervised 3D keypoint discovery.
- **images_synthesis** — synthetic data generation for pose.
- **fine-tuning_object_detection** — SSD fine-tuning workflow (largely covered by
  `object_detection/train.py`).
- **tutorials** — bounding boxes, augmentation, controlmap, detection pipeline.
- **Hand**: `ClassifyHandClosure` (open/closed hand) not ported.

### Remaining training scripts (cross-cutting)
Inference is ported for these, but no train.py yet: **HigherHRNet** (heatmap +
associative-embedding losses, COCO loader), **SimpleBaselines** (2D→3D lift,
H36M loader), **Minimal Hand** (DetNet/IKNet keypoint + IK losses),
**EfficientDet** (focal + smooth-L1, anchor matching).

## Implementation notes (selected)
- **EfficientPose:** model reuses the EfficientDet backbone/blocks + pose head;
  the 6D transformation/ADD loss is `paz.losses.MultiPoseLoss` (jit-friendly
  fixed-K positive selection). Box loss reused from `paz.losses.multibox`.
- **Human pose 3D 6D stage:** `paz.backend.poses` adds `project_to_image`,
  bone-ratio translation init, `scipy.least_squares` reprojection solve, and
  `human_pose3D_to_pose6D`; `EstimateHumanPose(solver, camera_intrinsics)` runs
  HigherHRNet → SimpleBaseline → optimize → 6D and draws the pose axes.
  `examples/human_pose_estimation_3D/viz.py` has matplotlib `show3Dpose`.
- **HigherHRNet:** bit-exact vs master; multi-person decode in
  `paz/backend/heatmaps.py` (scipy NMS + `linear_sum_assignment` grouping).
  `with_flip` TTA omitted (accuracy-only).
- **Action scores:** `ScalarActionScore` + `FeatureExtractor` Keras-3 callbacks
  in `examples/action_scores/`.
- **Probabilistic keypoints:** the legacy model embedded a
  `tensorflow_probability` mixture via `DistributionLambda`; the JAX port keeps
  the net convolutional (emits raw mixture maps) and moves the GMM math (mean,
  density, NLL) to `paz.backend.gaussian_mixture` + `paz.losses`
  `gaussian_mixture_nll`. `GMMKeypointNet2D` / `DetectGMMKeypointNet2D` apps;
  `examples/probabilistic_keypoint_estimation/` has the Kaggle loader, NLL
  `train.py`, and demo. No weights hosted — train to produce them.

## Locked decisions
- Weights hosting: `oarriaga/altamira-data`; we produce + verify
  `*_paz_jax.weights.h5`, the maintainer uploads, then URLs are wired.
- Isolation: git worktree on branch `port/domain-models` off `paz-jax`;
  verification on CPU; fast-forward `paz-jax` after each stage.

## Coordination with the foundation-models effort
- **Owned here:** `paz/models/{keypoint,pose_estimation,detection,segmentation}`,
  `paz/applications/*`, `paz/datasets/*`, `examples/*` (non-foundation),
  `docs/porting_roadmap.md`.
- **Owned by foundation (do not touch):** `paz/models/foundation/**`,
  `examples/{gemma3,gemma4,speech_to_text}`.
- **Shared registries** (`paz/models/__init__.py`, `paz/applications/__init__.py`,
  `paz/__init__.py`, `pyproject.toml`): edit append-only to keep merges trivial.
- **GPU:** pin domain verification (incl. TF reference subprocesses) to CPU,
  leaving the main GPU free for foundation models.
