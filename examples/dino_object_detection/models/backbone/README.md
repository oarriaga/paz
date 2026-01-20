# Backbone & Feature Projection Module

**Path:** `examples/dino_object_detection/models/backbone/`

This module is responsible for the feature extraction stage of the DINO / LW-DETR architecture. It adapts pre-trained Vision Transformer (ViT) foundation models (specifically DINOv2) to serve as backbones for object detection and generates multi-scale feature pyramids.

## Core Components

### 1. DINOv2 Backbone Wrapper

**File:** `dinov2_backbone_wrapper.py`

* **Purpose:** Wraps Keras-native ViT models (e.g., `vit_small`, `vit_large`) to expose intermediate feature maps required for detection.
* **Functionality:**
* Extracts features from specific transformer layers based on `out_feature_indexes`.
* Handles `PositionEmbeddingSine` generation for spatial awareness.
* Manages "Joiner" logic that combines the raw backbone output with position encodings.



### 2. Feature Projectors

**File:** `projector.py`

* **Purpose:** Projects the flat or single-scale output of the ViT backbone into a multi-scale Feature Pyramid Network (FPN) format required by the Transformer Decoder.
* **Implementations:**
* `SimpleProjector`: Basic projection using Convolutional blocks.
* `MultiScaleProjector`: Complex projector that generates features at varying resolutions (P3, P4, P5, P6) using depthwise-separable convolutions and upsampling/pooling.



## Testing & Parity

This module enforces strict numerical parity with the original PyTorch implementation.

* **Production Files:** `*_wrapper.py`, `projector.py` (Keras 3 / Backend-agnostic).
* **Reference Files:** Files prefixed with `torch_` (e.g., `torch_backbone_for_testing.py`) contain the original PyTorch logic strictly for verification.
* **Tests:**
* `backbone_test.py`: Verifies that the Keras ViT wrapper outputs match the PyTorch reference for specific input seeds.
* `projector_test.py`: Ensures the multi-scale projection math (convolutions, normalization, upsampling) is identical between frameworks.



## File Manifest

| File | Description |
| --- | --- |
| **`dinov2_backbone_wrapper.py`** | Main Keras wrapper for ViT backbones and Position Embeddings. |
| **`projector.py`** | Keras implementation of Simple and MultiScale feature projectors. |
| `backbone_test.py` | Unit tests comparing Keras backbone outputs against PyTorch. |
| `projector_test.py` | Unit tests comparing Keras projector outputs against PyTorch. |
| `torch_backbone_for_testing.py` | Reference PyTorch backbone logic (frozen). |
| `torch_projector_for_testing.py` | Reference PyTorch projector logic (frozen). |
| `torch_position_encoding...` | Reference PyTorch position embedding logic (frozen). |