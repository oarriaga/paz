# Segmentation & Panoptic Head

**Path:** `examples/dino_object_detection/models/segmentation_head/`

This module implements the **Panoptic Segmentation Head** for the DINO / LW-DETR architecture. It extends the object detector to predict pixel-wise binary masks for every detected object. It employs a lightweight, attention-style mechanism where object queries (from the Transformer) are dot-produced with a high-resolution feature map to generate masks.

## Core Components

### 1. Segmentation Head

**File:** `segmentation_head.py`

* **Purpose:** The top-level module that generates mask logits.
* **Architecture:**
* **Spatial Branch:** Takes the high-resolution feature map (e.g., from the backbone or FPN) and refines it using a stack of `DepthwiseConvBlock` layers.
* **Query Branch:** Takes the object embeddings (queries) from the Transformer Decoder and projects them using an `MLPBlock`.
* **Fusion:** Computes the dot product (via `einsum`) between the projected queries and the spatial features to create unique masks for each object.



### 2. Building Blocks

* **`depthwise_conv_block.py`**: A modernized convolutional block used in the spatial branch. It features Depthwise Separable Convolutions, Layer Normalization (with channel-first support), GELU activation, and **LayerScale** for training stability.
* **`mlp_block.py`**: A Multi-Layer Perceptron used to project object queries into the same embedding space as the spatial features. Also utilizes LayerScale.

### 3. Point Sampling Utilities

**File:** `utils.py`

* **Purpose:** Implements the "PointRend" style efficiency logic. Instead of computing the loss on the entire high-resolution mask (which is memory expensive), the model samples specific points during training.
* **Key Functions:**
* **`point_sample`**: Performs bilinear interpolation to sample features at arbitrary continuous coordinates.
* **`get_uncertain_point_coords_with_randomness`**: Selects the most "uncertain" points (boundary regions where logits are close to 0) to focus the loss calculation where it matters most.



## Testing & Parity

This module enforces strict numerical parity with the original PyTorch implementation to ensure the mask branch trains correctly.

* **Tests:**
* `segmentation_head_test.py`: End-to-end parity check of the `SegmentationHead` forward pass (logits match PyTorch ).
* `utils_test.py`: Verifies `point_sample` numerical accuracy and uncertainty sampling logic.
* `depthwise_conv_block_test.py` / `mlp_block_test.py`: Unit tests for individual layers.


* **Reference:** `torch_segmentation_head_for_testing.py` contains the frozen PyTorch logic used as the ground truth.

## File Manifest

| File | Description |
| --- | --- |
| **`segmentation_head.py`** | Main class `SegmentationHead`. Fuses spatial maps and object queries. |
| **`utils.py`** | `point_sample` and uncertainty-based coordinate selection. |
| **`depthwise_conv_block.py`** | LayerScale-enabled Depthwise Conv block for spatial features. |
| **`mlp_block.py`** | LayerScale-enabled MLP for query features. |
| `segmentation_head_test.py` | Parity test for the full head. |
| `utils_test.py` | Unit tests for sampling logic. |
| `*_block_test.py` | Unit tests for building blocks. |
| `torch_segmentation_head...` | Reference PyTorch implementation (frozen). |