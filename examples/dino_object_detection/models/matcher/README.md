# Hungarian Matcher Module

**Path:** `examples/dino_object_detection/models/matcher/`

This module implements the **Hungarian Matcher** (bipartite graph matching) algorithm, which is the core assignment mechanism for DETR-based object detectors. Unlike traditional detectors that use fixed anchors and NMS, DINO/LW-DETR uses this matcher to force a one-to-one mapping between the model's predictions and the ground-truth objects during training.

## Core Components

### 1. Keras Hungarian Matcher

**File:** `matcher.py`

* **Purpose:** Computes the optimal assignment between a set of  predictions (objects queries) and  ground-truth objects () to minimize the total matching cost.
* **Cost Function:** The cost matrix is a weighted sum of:
* **Classification Cost:** Sigmoid Focal Loss (probability of the correct class).
* **Box L1 Cost:** L1 distance between predicted and target box coordinates.
* **GIoU Cost:** Generalized Intersection over Union (handles non-overlapping boxes).
* **Mask Cost:** (Optional) Combination of Focal and Dice loss for segmentation masks.


* **Implementation Details:**
* Since the Hungarian algorithm is non-differentiable and combinatorially complex, the actual solving step uses `scipy.optimize.linear_sum_assignment` on the CPU.
* It handles batching by iterating over batch elements or using `group_detr` logic (for hybrid-group training).
* Inputs/Outputs are handled as Keras tensors, but converted to Numpy for the SciPy solver.



### 2. Parity Testing

**File:** `matcher_test.py`

* **Purpose:** Verifies that the Keras implementation produces **identical** assignments to the original PyTorch code for the same set of inputs.
* **Methodology:**
* Generates random "predictions" (logits, boxes, masks) and "targets".
* Runs the `TorchMatcher` (reference) and `KerasMatcher` side-by-side.
* Asserts that the resulting indices (permutation of prediction indices) are exactly the same.



### 3. PyTorch Reference

**File:** `torch_matcher_for_testing.py`

* **Purpose:** A frozen copy of the original PyTorch Hungarian Matcher.
* **Usage:** Used strictly as the "Ground Truth" for unit tests to ensure the cost calculation logic (especially with Focal Loss and GIoU) is mathematically consistent with the Keras version.

## File Manifest

| File | Description |
| --- | --- |
| **`matcher.py`** | **Main Keras Implementation.** `HungarianMatcher` class. Uses SciPy for assignment. |
| `matcher_test.py` | Unit tests ensuring Keras assignment indices match PyTorch exactly. |
| `torch_matcher_for_testing.py` | Reference PyTorch implementation (frozen) used only for parity verification. |