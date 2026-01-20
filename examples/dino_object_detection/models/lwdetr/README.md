# LW-DETR / DINO Model Assembly

**Path:** `examples/dino_object_detection/models/lwdetr/`

This directory contains the high-level architecture definition for the DINO (LW-DETR) Object Detector. It is where the distinct components (Backbone, Transformer, Segmentation Heads, and Matchers) are assembled into a cohesive, trainable Keras model.

## Core Components

### 1. Keras Model Assembly

**File:** `lwdetr_keras.py`

* **Purpose:** The production-ready Keras 3 implementation of the full detection pipeline.
* **Key Functionality:**
* **`build_model(args)`:** The primary entry point. It instantiates the backbone (ViT/DINOv2), the Transformer Decoder, and the Heads, linking them into a single `keras.Model`.
* **Criterion & Loss:** Defines the `SetCriterion` logic, which computes the Multi-Task Loss (Classification + Box Regression + GIoU + Mask) after performing Hungarian Matching.
* **Post-Processing:** Includes logic to convert raw model outputs (logits/box deltas) into final detection formats (scores, labels, bounding boxes).



### 2. Parity Testing

**File:** `lwdetr_test.py`

* **Purpose:** An end-to-end integration test ensuring the Keras implementation behaves exactly like the original PyTorch code.
* **Workflow:**
1. Builds the PyTorch reference model (`torch_lwdetr_for_testing.py`).
2. Builds the Keras model (`lwdetr_keras.py`).
3. **Weight Porting:** Manually maps and transfers every weight parameter from PyTorch to Keras (handling Shape permutations for Convolutions and Dense layers).
4. **Inference:** Runs both models on identical random noise input.
5. **Assertion:** Verifies that final outputs (Class Logits, Box Coordinates, Mask Maps) match within strict floating-point tolerances ().



### 3. PyTorch Reference

**File:** `torch_lwdetr_for_testing.py`

* **Purpose:** A frozen copy of the original LW-DETR/DINO PyTorch source code.
* **Usage:** It is **never** used in the production Keras pipeline. It exists solely as the "Ground Truth" for the unit tests in `lwdetr_test.py` to validate mathematical correctness.

## File Manifest

| File | Description |
| --- | --- |
| **`lwdetr_keras.py`** | **Main implementation.** Contains `DINO` class, `PostProcess`, `SetCriterion`, and `build_model`. |
| `lwdetr_test.py` | End-to-end test suite comparing Keras outputs vs. PyTorch outputs. |
| `torch_lwdetr_for_testing.py` | Reference PyTorch implementation (frozen) used only for verification. |