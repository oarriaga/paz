# Transformer Decoder & Head Module

**Path:** `examples/dino_object_detection/models/transformer_decoder_head/`

This module implements the core **Transformer Decoder** logic for the DINO / LW-DETR architecture. It is responsible for taking the multi-scale feature maps from the backbone/projector and refining a set of learnable "Object Queries" into final object detections using attention mechanisms.

## Core Components

### 1. Keras Transformer Decoder

**File:** `transformer_kerass.py`

* **Purpose:** The primary Keras 3 implementation of the DINO Transformer Decoder.
* **Key Features:**
* **`Transformer` Class:** The top-level container that manages the sequence of decoder layers.
* **Coordinate embeddings:** Implements `gen_sineembed_for_position` to generate high-frequency sine/cosine embeddings from reference point coordinates (normalized 0-1).
* **Query Updates:** Handles the iterative refinement of object queries (hidden states) and their corresponding reference points (boxes) across multiple decoder layers.
* **Cross-Attention:** Mechanisms to attend to the multi-scale feature maps provided by the backbone.



### 2. Parity Testing

**File:** `transformer_test.py`

* **Purpose:** Ensures the Keras implementation is mathematically identical to the original PyTorch code.
* **Methodology:**
* Instantiates both the Keras `Transformer` and the PyTorch `Transformer` (from `torch_transformer_for_testing.py`).
* Feeds identical random inputs (Source Features, Masks, Position Embeddings, Query Embeddings).
* Transfers weights layer-by-layer.
* Asserts that outputs (Hidden States, Reference Points) match within floating-point tolerance ().



### 3. PyTorch Reference

**File:** `torch_transformer_for_testing.py`

* **Purpose:** A frozen copy of the original PyTorch Transformer implementation.
* **Usage:** Used strictly as a "Ground Truth" for unit tests. It is not used in production inference or training.

## File Manifest

| File | Description |
| --- | --- |
| **`transformer_kerass.py`** | **Main Keras Implementation.** Contains the `Transformer` class and `gen_sineembed_for_position` logic. |
| `transformer_test.py` | Unit tests verifying that Keras decoder outputs match PyTorch exactly. |
| `torch_transformer_for_testing.py` | Reference PyTorch implementation (frozen) used only for verification. |