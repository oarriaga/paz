# DINOv3 Keras Porting - Project Overview & Usage

This project implements the DINOv3 architecture (ViT and ConvNeXt) using **Keras 3** (with JAX backend), intended to match the official PyTorch implementation numerically.

## 1\. Project Structure

Based on the current directory layout, here is where the key components reside:

```text
PAZ/paz/models/foundation/dinov3/
├── dinov3_test.py                              # Master integration test 
├── port_dino_weights_from_torch_to_keras.py    # Script to convert .pth -> .keras
├── utils/
│   ├── utils.py                                # Helper functions
│   └── ...
├── layers/                                     # Layers Implementations & Unit Tests
│   ├── attention.py / attention_test.py
│   ├── block.py / block_test.py
│   ├── ffn_layers.py / ffn_layers_test.py
│   ├── layer_scale.py / layer_scale_test.py
│   ├── patch_embed.py / patch_embed_test.py
│   ├── rms_norm.py / rms_norm_test.py
│   ├── rope_position_encoding.py / rope_position_encoding_test.py
│   └── torch_layers_for_testing.py             # PyTorch reference layers
└── models/                                     # Full Model Architectures
    ├── convnext.py                             # Keras ConvNeXt
    ├── vision_transformer.py                   # Keras ViT
    ├── torch_convnext_for_testing.py           # PyTorch ConvNeXt wrapper
    ├── torch_vision_transformer_for_testing.py # PyTorch ViT wrapper
    ├── convnext_test.py                        # ConvNeXt model verification
    └── vision_transformer_test.py              # ViT model verification
```

-----

## 2\. Environment Setup

This project requires **Python 3.10+**. We rely on Keras 3 with JAX as the backend, and PyTorch for reference comparison.

### Installation
1. **Create Environment**
    ```bash 
    conda create --name paz_dev_env python=3.10
    conda activate paz_dev_env
    ```


2.  **Install PyTorch:**

    ```bash
    pip install torch torchvision
    ```

3.  **Install Keras 3 and JAX:**

    ```bash
    pip install "keras>=3.0.0"
    pip install "jax<0.7.0" "jaxlib<0.7.0"
    ```

4.  **Install testing Dependencies:**

    ```bash
    pip install pytest numpy
    ```

5. **Install other Dependencies: (neede by paz somewhere else e.g. (paz\backend\plane.py))**
    ```bash
    pip install matplotlib opencv-python optax tensorflow_probability
    ```

6. **Make the Keras Backend to jax**
    ```bash 
    $env:KERAS_BACKEND = "jax"
    $env:PYTHONPATH = "path/to/your/paz"
-----

## 3\. Configuration & Weight Preparation

Before running full model tests, you must download the official DINOv3 PyTorch weights and configure the scripts to find them.

### Step A: Download Weights

Download the `.pth` checkpoints for the models you wish to test (e.g., `dinov3_vits16`, `dinov3_convnext_tiny`) and place them in a known directory.

### Step B: Update Path Configurations

You must update the hardcoded paths in the testing scripts to match your local machine.

1.  **Open** `dinov3_test.py` and `port_dino_weights_from_torch_to_keras.py`.
2.  **Locate** the configuration section (usually near the top):
    ```python
    DINO_REPO_PATH = r"path/to/your/dinov3_repo" 
    PT_MODELS_WEIGHTS_DIR_PATH = r"path/to/directory/containing/pth_files"
    ```
3.  **Update** these strings to point to your actual local directories.

-----

## 4\. Workflow: Porting Weights

Before running the master test (`dinov3_test.py`), you must convert the PyTorch weights (`.pth`) into Keras weights (`.keras`).

Run the porting script:

```bash
paz/models/foundation/dinov3/port_dino_weights_from_torch_to_keras.py
```

**What this does:**

1.  Loads the PyTorch definition and weights.
2.  Instantiates the Keras architecture.
3.  Transfers weights layer-by-layer.
4.  Saves the resulting models to a local folder (default: `weights_dinov3/`).

-----

## 5\. Running Tests

You can run tests at three different levels: atomic layers, specific models, or the full integration suite.

### Level 1: Unit Tests (Layers)

Test individual components (Attention, RoPE, LayerScale, etc.) to ensure they function correctly in isolation.

```bash
# Run all layer tests
pytest paz/models/foundation/dinov3/layers/

# Or run a specific layer test
pytest paz/models/foundation/dinov3/layers/block_test.py
pytest paz/models/foundation/dinov3/layers/attention_test.py
. 
.
.

```

### Level 2: Model Verification

Test the specific architectures to ensure the graph is built correctly and weights can be loaded.

**Vision Transformer (ViT):**

```bash
pytest paz/models/foundation/dinov3/models/vision_transformer_test.py
```

**ConvNeXt:**

```bash
pytest paz/models/foundation/dinov3/models/convnext_test.py
```

*(Note: Ensure `MODEL_PARAMS` in `convnext_test.py` points to your valid weight files).*

### Level 3: Master Integration Test

This is the comprehensive test of the dinov3 model deeply using the accual weights that validates the ported `.keras` models against the PyTorch references.

```bash
pytest paz/models/foundation/dinov3/dinov3_test.py
```
