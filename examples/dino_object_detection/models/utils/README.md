# Utilities & Shared Infrastructure

**Path:** `examples/dino_object_detection/models/utils/`

This directory contains the foundational building blocks, geometric operations, training infrastructure, and benchmarking tools required by the DINO / LW-DETR object detector. It acts as the "standard library" for the project, ensuring consistent behavior across the backbone, transformer, and heads.

## Core Components

### 1. Geometric & Box Operations

**File:** `box_ops.py`

* **Purpose:** Implements vectorizable bounding box manipulations compatible with Keras backend tensors.
* **Key Functions:**
* `box_cxcywh_to_xyxy` / `box_xyxy_to_cxcywh`: Coordinate format conversions.
* `box_iou`: Standard Intersection over Union.
* `generalized_box_iou`: GIoU implementation (crucial for DETR-based loss).
* `box_area`: Computes area for box tensors.



### 2. Training Infrastructure

**Files:** `get_param_dicts.py`, `drop_scheduler.py`, `early_stopping.py`, `utils.py`

* **Parameter Groups (`get_param_dicts.py`):** Implements **Layer-wise Learning Rate Decay**, a critical technique for training ViT-based detectors. It groups model parameters and assigns decaying learning rates based on their depth in the transformer backbone.
* **Drop Path Scheduler (`drop_scheduler.py`):** Manages the stochastic depth rate over training epochs, often increasing the drop rate linearly.
* **Early Stopping (`early_stopping.py`):** A custom callback that monitors mAP (bbox or mask) and halts training when performance plateaus.
* **Model EMA (`utils.py`):** Implements Exponential Moving Average for model weights, which typically yields better inference performance than the final raw weights.

### 3. Benchmarking & Metrics

**Files:** `benchmark.py`, `metrics.py`

* **FLOPs & FPS (`benchmark.py`):** A tool to measure the theoretical computational cost (FLOPs) and practical throughput (FPS) of the Keras model. It handles backend-specific synchronization to ensure accurate timing.
* **Logging (`metrics.py`):** Utilities for logging training progress to **Weights & Biases (WandB)** and generating local matplotlib plots for loss curves and mAP scores.

### 4. General Utilities

**Files:** `misc.py`, `files.py`, `coco_classes.py`

* **NestedTensor (`misc.py`):** A Python shim to handle batches of images with varying sizes (padding masks) within Keras, mimicking PyTorch's `NestedTensor` behavior.
* **SmoothedValue (`misc.py`):** Tracks running averages (median, global avg) for loss logging.
* **Asset Management (`files.py`):** Helper functions to download pretrained weights or config files.
* **Data Mapping (`coco_classes.py`):** Dictionary mapping COCO class indices to human-readable names.
* **Weight Transfer (`obj365_to_coco_model.py`):** Utilities for adapting models pre-trained on Objects365 to the COCO dataset (filtering class weights).

## Parity Testing

This module ensures that low-level operations (like box IoU or parameter grouping) match the reference implementation exactly.

* **Tests:** `box_ops_test.py`, `misc_test.py`, `benchmark_test.py`, `get_param_dicts_test.py`.
* **Reference:** Files prefixed with `torch_` (e.g., `torch_box_ops_for_testing.py`) contain the original PyTorch logic for verification.

## File Manifest

| File | Description |
| --- | --- |
| **`box_ops.py`** | Keras implementation of IoU, GIoU, and coordinate transforms. |
| **`benchmark.py`** | Utilities for counting FLOPs and measuring inference FPS. |
| **`get_param_dicts.py`** | Logic for layer-wise learning rate decay and parameter grouping. |
| **`metrics.py`** | WandB logging integration and metric plotting. |
| **`misc.py`** | `NestedTensor` abstraction and numerical helpers (`SmoothedValue`). |
| **`utils.py`** | `ModelEMA` (Exponential Moving Average) implementation. |
| `early_stopping.py` | Custom callback for early stopping based on mAP. |
| `drop_scheduler.py` | Scheduler for Stochastic Depth (DropPath) rates. |
| `coco_classes.py` | ID-to-Label mapping for the COCO dataset. |
| `files.py` | HTTP file download helper. |
| `obj365_to_coco_model.py` | Helper to fine-tune Obj365 models on COCO. |
| `*_test.py` | Unit tests for specific utility modules. |
| `torch_*_for_testing.py` | Reference PyTorch implementations used strictly for parity checks. |