# paz porting roadmap (legacy `master` → JAX/Keras-3)

Living document. The new `paz` is a JAX + Keras-3 perception library spanning
domain-specific perception (past), foundation models (present), and
probabilistic graphics (future). This file tracks which **domain-specific**
models from the legacy `master` branch we recover, in what order, and the state
of their ported weights. A parallel effort ports foundation models; see the
partition at the bottom.

Update the status table as work lands. Keep it terse.

## Conventions for every port
- Inference-only (no training scripts) unless stated.
- Clean functional style: models are functions returning a Keras `Model`;
  applications are functions with inner `preprocess`/`postprocess`/`call`
  closures. No `Processor`/`Pipeline`/`SequentialProcessor` classes.
- Follow `CLAUDE.md` (80 cols, no type hints, no docstrings on small helpers,
  `build_*`/`compute_*`, `box2D`/`box3D`, namedtuples for structured data).
- Weights: legacy `.hdf5` live at `github.com/oarriaga/altamira-data/releases`.
  Each port produces a Keras-3 `*_paz_jax.weights.h5` + a numeric parity test
  vs a TF reference (TF pinned to CPU). The new file is uploaded to a new
  altamira-data release; then `WEIGHT_PATH` is repointed.
- Reference patterns: `paz/applications/detectors.py` (app shape),
  `paz/models/segmentation/unet.py` (model-as-function),
  `paz/models/keypoint/{detnet,iknet}.py` + `weights_test.py` (weights+parity).

## Ranked porting list

### Tier 1 — core robot-vision domain capabilities
| Rank | Model | Task | Legacy weights | Status |
|------|-------|------|----------------|--------|
| 1 | Minimal Hand (DetNet + IKNet) | single-hand 2D/3D keypoints + joint angles | v0.14 detnet/iknet | ✅ done |
| 2 | HigherHRNet | 2D multi-person human pose | v0.10 | ✅ done |
| 3 | SimpleBaselines | 3D human pose (2D→3D lift) | v0.17 | ✅ done |
| 4 | EfficientPose | 6D object pose (LINEMOD) | altamira-data | in progress |
| 5 | Pix2Pose | 6D object pose (YCB / power drill) | altamira-data | backlog |

### Tier 2 — high value, partially started
| Rank | Model | Task | Legacy weights | Status |
|------|-------|------|----------------|--------|
| 6 | EfficientDet D0–D7 | object detection (COCO) | v0.16 | ✅ done (COCO); VOC variant deferred |
| 7 | Mask R-CNN | instance segmentation | altamira-data | backlog |

### Already ported (no action)
SSD300-VOC, SSD512-COCO, Haar cascade, MiniXception (emotion FER), face
KeypointNet2D32, HeadPose, STN, ProtoNet, Xception, PCA, kNN/DBSCAN, MAML,
eigenfaces.

### Lower priority / niche
SSD512 hand-detection, SSD512-YCBVideo, SSD300-FAT, UNet pretrained pipeline,
VVAD (visual voice activity, cnn2Plus1).

## Training scripts (cross-cutting workstream)
All ports so far are **inference-only** (load pretrained weights, run, draw).
A parallel workstream is to add **training scripts** so these models can be
retrained/fine-tuned in the JAX/Keras-3 stack. Per model this means: dataset
loader, data augmentation pipeline, loss(es), optimizer + schedule, and a
`train.py`. Prioritize where retraining is most useful:
- EfficientPose (LINEMOD 6D) — losses + LINEMOD loader + train loop.
- EfficientDet / SSD — detection losses (focal + smooth-L1), anchors matching.
- HigherHRNet / SimpleBaselines — heatmap/AE losses, H36M/COCO loaders.
- Hand (DetNet/IKNet) — keypoint + IK losses.
Track each model's training status alongside its inference status as it lands.

## Detailed status (active work)
All four load ported Keras-3 native `*_paz_jax.weights.h5` assets, verified by
downloading fresh from the releases (sha256 match + load + run):

| Model | Model code | Native weights | Parity test | Application | Done |
|-------|-----------|----------------|-------------|-------------|------|
| DetNet | ✅ | v0.14 ✅ verified | ✅ round-trip | ✅ | ✅ |
| IKNet | ✅ | v0.14 ✅ verified | ✅ round-trip | ✅ | ✅ |
| HigherHRNet | ✅ | v0.10 ✅ verified | ✅ bit-exact | ✅ | ✅ |
| SimpleBaselines | ✅ | v0.17 ✅ verified | ✅ bit-exact | ✅ | ✅ |

### SimpleBaselines notes
- Clean Keras-3 MLP, **bit-exact** vs the master (0.0 diff). Loads the ported
  native `v0.17/simple_baseline_paz_jax.weights.h5` via `model.load_weights`.
- Human3.6M normalization constants copied verbatim into
  `paz/datasets/human36m.py`.
- Applications `EstimateHumanPose3D` (2D→3D lift) and `EstimateHumanPose`
  (HigherHRNet 2D → SimpleBaseline 3D) in `human_pose_estimators.py`. Verified
  end-to-end: produces an anatomically-correct 3D skeleton. The master's
  `EstimateHumanPose` additionally runs a 6D-pose optimization (camera/solver)
  that is out of scope for the lift and deferred.

### HigherHRNet notes
- Clean Keras-3 port is **bit-exact** vs the master model (0.0 diff on both
  outputs), verified against the master code run as an independent reference.
- Loads the ported native `v0.10/higher_hrnet_paz_jax.weights.h5` via
  `model.load_weights`. (The native file was regenerated from the fixed model;
  an earlier upload saved before the fuse fix did not native-load and was
  replaced.)
- Multi-person postprocessing ported as clean functional numpy in
  `paz/backend/heatmaps.py` (NMS via `scipy.ndimage.maximum_filter`,
  associative-embedding grouping via `scipy.optimize.linear_sum_assignment`),
  with the `HigherHRNetHumanPose2D` application in
  `paz/applications/human_pose_estimators.py`.
- Verified end-to-end on COCO images: correct single-person and 4-person
  grouped skeletons. The master postprocessing depends on the old-paz
  framework (not runnable here), so postprocessing is verified structurally +
  visually + unit-tested on the grouping, not bit-exact.
- `with_flip` test-time augmentation is omitted (single tag channel); it is an
  accuracy refinement, not required for correct multi-person output.

### EfficientDet notes
- Model + anchors were already ported and load v0.16 COCO weights. Added the
  clean detection applications `EFFICIENTDETD0COCO`..`EFFICIENTDETD7COCO` in
  `paz/applications/detectors.py`: ImageNet-normalize + aspect-preserving
  `scaled_resize`, `change_box_coordinates`, decode with `model.prior_boxes`
  (variances `[1,1,1,1]`), scale boxes back, per-class NMS — reusing the
  existing `paz.detection` helpers. Added `COCO_EFFICIENTDET` (90) labels.
- Verified end-to-end on COCO images (D0 and D1): correct, well-localized
  detections (people, cars, baseball bat/glove). The engine is size-agnostic
  (reads input size + priors from the model), so D0–D7 share one code path.
- VOC variant (`EFFICIENTDETD0VOC`) deferred — different class set/weights.

### Hand estimation weights
DetNet and IKNet load ported native `v0.14/{detnet,iknet}_paz_jax.weights.h5`
via `model.load_weights`, verified to round-trip exactly against the legacy
`.hdf5` (xyz/uv/quaternion match to 1e-6) and downloaded fresh from the release.

## Locked decisions
- Hand: **single-hand pose only** (`MinimalHandPoseEstimation`) — no hand
  detector, no open/close classification.
- HigherHRNet: **full multi-person** decoding (associative-embedding grouping
  via `scipy.optimize.linear_sum_assignment`, verified against legacy Munkres).
- Weights hosting: new `oarriaga/altamira-data` release; we produce + verify
  `*_paz_jax.weights.h5`, the maintainer uploads, then URLs are wired.
- Isolation: dedicated git worktree on branch `port/domain-models` off
  `paz-jax`; verification pinned to CPU / a free GPU.

## Coordination with the foundation-models effort
- **Owned by this effort:** `paz/models/{keypoint,pose_estimation,detection}`,
  `paz/applications/*`, `paz/datasets/*`, `examples/*` (non-foundation),
  `docs/porting_roadmap.md`.
- **Owned by the foundation effort (do not touch):**
  `paz/models/foundation/**`, `examples/{gemma3,gemma4,speech_to_text}`.
- **Shared registries** (`paz/models/__init__.py`,
  `paz/applications/__init__.py`, `paz/__init__.py`, `pyproject.toml`): edit
  append-only, one import line per model, to keep merges trivial.
- **GPU:** foundation models are memory-hungry; domain models are small. Pin
  domain verification (incl. TF reference subprocesses) to CPU
  (`CUDA_VISIBLE_DEVICES=""`) or a free device, leaving the main GPU free.
