import os
import sys
import logging
import torch
import numpy as np
import pytest

# --- Configuration for Keras Backend and Environment ---

os.environ["KERAS_BACKEND"] = "jax"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import keras

# ==============================================================================
# Keras Layer Implementation
# ==============================================================================
from paz.models.foundation.dinov3.models.vision_transformer import (
    vit_small,
    vit_base,
    vit_large,
)

# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================
from paz.models.foundation.dinov3.models.torch_vision_transformer_for_testing import (
    PT_vit_small,
    PT_vit_base,
    PT_vit_large,
)

# ==============================================================================
# Global Configuration
# ==============================================================================

DINO_REPO_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
PT_MODELS_WEIGHTS_DIR_PATH = (
    r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3/"
)
KERAS_MODELS_WEIGHTS_DIR_PATH = (
    r"weights_dinov3"
)

PT_MODEL_WEIGHTS_PATHS = {
    "dinov3_vits16": r"dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitb16": r"dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": r"dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}
KERAS_MODEL_WEIGHTS_PATHS = {
    "dinov3_vits16": r"dinov3_vits16_ported.keras",
    "dinov3_vitb16": r"dinov3_vitb16_ported.keras",
    "dinov3_vitl16": r"dinov3_vitl16_ported.keras",
}

model_kwargs = {
    "img_size": 224,
    "patch_size": 16,
    "ffn_layer": "mlp",
    "untie_cls_and_patch_norms": False,
    "norm_layer": "layernorm",
    "layerscale_init": 1e-6,
    "n_storage_tokens": 4,
    "pos_embed_rope_dtype": "float32",
}

# ==============================================================================
# Helper Functions
# ==============================================================================
MODEL_CONFIGS = [
    # (PyTorch Constructor, Keras Constructor, Weight Key)
    (PT_vit_small, vit_small, "dinov3_vits16"),
    (PT_vit_base, vit_base, "dinov3_vitb16"),
    (PT_vit_large, vit_large, "dinov3_vitl16"),
]


def get_PT_model(pt_constructor, pt_weight_path_key, model_kwargs):
    """Instantiates a PyTorch model and loads its pre-trained weights."""
    if DINO_REPO_PATH not in sys.path:
        sys.path.insert(0, DINO_REPO_PATH)

    pt_weight_path = os.path.join(
        PT_MODELS_WEIGHTS_DIR_PATH, PT_MODEL_WEIGHTS_PATHS[pt_weight_path_key]
    )
    if not os.path.isfile(pt_weight_path):
        pytest.skip(f"PyTorch weight file not found: {pt_weight_path}")

    pt_model = pt_constructor(**model_kwargs)
    state_dict = torch.load(pt_weight_path, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=False)
    pt_model.eval()
    print(f"PyTorch model '{pt_weight_path_key}' loaded from {pt_weight_path}")
    return pt_model


def get_keras_model(keras_constructor, keras_weight_path_key, model_kwargs):
    """Instantiates a Keras model, builds it, and loads its ported weights."""
    keras_weight_path = os.path.join(
        KERAS_MODELS_WEIGHTS_DIR_PATH,
        KERAS_MODEL_WEIGHTS_PATHS[keras_weight_path_key],
    )
    if not os.path.isfile(keras_weight_path):
        pytest.skip(f"Keras weight file not found: {keras_weight_path}")

    keras_model = keras_constructor(**model_kwargs)

    # Build the model by passing a dummy input
    dummy_input = np.random.randn(
        1, model_kwargs["img_size"], model_kwargs["img_size"], 3
    ).astype("float32")
    keras_model(dummy_input, training=False)

    # Load the pre-ported weights
    keras_model.load_weights(keras_weight_path)
    print(f"Keras model '{keras_weight_path_key}' loaded from {keras_weight_path}")
    return keras_model


# ==============================================================================
# Test Suite for Final Output
# ==============================================================================

@pytest.mark.parametrize("pt_constructor, keras_constructor, weight_key", MODEL_CONFIGS)
def test_final_output_equivalence(pt_constructor, keras_constructor, weight_key):
    print(f"\n--- Testing Final Output Equivalence: {weight_key} ---")
    pt_model = get_PT_model(pt_constructor, weight_key, model_kwargs)
    keras_model = get_keras_model(keras_constructor, weight_key, model_kwargs)

    # --- Create Identical Inputs ---
    np.random.seed(42)  # Use a fixed seed for reproducibility
    dummy_input_np = np.random.randn(1, 224, 224, 3).astype("float32")
    keras_input = keras.ops.convert_to_tensor(dummy_input_np)
    pt_input = torch.from_numpy(dummy_input_np.transpose(0, 3, 1, 2))
    print("Generated identical inputs.")

    # --- Get Model Outputs ---
    print("Performing inference...")
    keras_output = keras_model(keras_input, training=False)
    with torch.no_grad():
        pt_output = pt_model(pt_input, is_training=False)

    # --- Compare Outputs ---
    print("Comparing outputs...")
    keras_output_np = keras.ops.convert_to_numpy(keras_output)
    pt_output_np = pt_output.detach().cpu().numpy()

    try:
        np.testing.assert_allclose(
            keras_output_np,
            pt_output_np,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Final model outputs do not match after loading weights",
        )
    except AssertionError as e:
        print("\n❌ Outputs do not match!")
        print(
            f"Max absolute difference: {np.max(np.abs(keras_output_np - pt_output_np))}"
        )
        print(
            f"Mean absolute difference: {np.mean(np.abs(keras_output_np - pt_output_np))}"
        )
        raise e
    print(f"✅ Success! Final outputs for {weight_key} match.")


# ==============================================================================
# Test Suite for Deep-Dive Block Equivalence
# ==============================================================================


@pytest.mark.parametrize("pt_constructor, keras_constructor, weight_key", MODEL_CONFIGS)
def test_deep_dive_block_equivalence(pt_constructor, keras_constructor, weight_key):
    print(f"\n--- Testing Deep-Dive Block Equivalence: {weight_key} ---")
    pt_model = get_PT_model(pt_constructor, weight_key, model_kwargs)
    keras_model = get_keras_model(keras_constructor, weight_key, model_kwargs)

    any_failure = False

    def _compare_and_correct(keras_tensor, torch_tensor, layer_name, block_idx):
        nonlocal any_failure
        torch_np = torch_tensor.detach().cpu().numpy()
        keras_np = keras.ops.convert_to_numpy(keras_tensor)
        try:
            mean_diff = np.mean(np.abs(keras_np - torch_np))
            print(
                f"    Comparing Block {block_idx} - {layer_name}: Mean abs diff = {mean_diff}"
            )
            if np.isnan(keras_np).any():
                raise AssertionError("Keras output contains NaN values.")
            if mean_diff > 1e-4:
                raise AssertionError(
                    f"Mean absolute difference {mean_diff} exceeds tolerance of 1e-4."
                )
            print(f"    ✅ Block {block_idx} - {layer_name}: Output matches.")
            return keras_tensor
        except AssertionError as e:
            print(
                f"    ❌ FAILURE at Block {block_idx} - {layer_name}: Output mismatch."
            )
            print(f"       {e}")
            any_failure = True
            print(f"       Correcting Keras tensor to continue analysis...")
            return keras.ops.convert_to_tensor(torch_np)

    for i in range(len(keras_model.blocks)):
        print(f"\n--- Analyzing Block {i} ---")

        # 1. Create a new, unique input for this block to isolate errors
        np.random.seed(i)
        image_np = np.random.randn(
            1, model_kwargs["img_size"], model_kwargs["img_size"], 3
        ).astype("float32")
        keras_input = keras.ops.convert_to_tensor(image_np)
        pt_input = torch.from_numpy(image_np.transpose(0, 3, 1, 2))

        # 2. Get initial tokens
        keras_tokens, (H_k, W_k) = keras_model.prepare_tokens_with_masks(keras_input)
        pt_tokens, (H_pt, W_pt) = pt_model.prepare_tokens_with_masks(pt_input)

        # 3. Get RoPE embeddings
        rope_keras = keras_model.rope_embed(H=H_k, W=W_k, training=False)
        rope_torch = pt_model.rope_embed(H=H_pt, W=W_pt)

        # 4. Get the corresponding block from each model
        keras_block = keras_model.blocks[i]
        pt_block = pt_model.blocks[i]

        # --- Deep Dive into the Block ---
        keras_norm1_out = keras_block.norm1(keras_tokens)
        pt_norm1_out = pt_block.norm1(pt_tokens)
        keras_norm1_out = _compare_and_correct(
            keras_norm1_out, pt_norm1_out, "Norm1", i
        )

        keras_attn_out = keras_block.attn(
            keras_norm1_out, rope=rope_keras, training=False
        )
        pt_attn_out = pt_block.attn(pt_norm1_out, rope=rope_torch)
        keras_attn_out = _compare_and_correct(
            keras_attn_out, pt_attn_out, "Attention", i
        )

        keras_ls1_out = keras_block.ls1(keras_attn_out, training=False)
        pt_ls1_out = pt_block.ls1(pt_attn_out)
        keras_ls1_out = _compare_and_correct(
            keras_ls1_out, pt_ls1_out, "LayerScale1", i
        )

        keras_res1_out = keras_tokens + keras_ls1_out
        pt_res1_out = pt_tokens + pt_ls1_out
        keras_res1_out = _compare_and_correct(
            keras_res1_out, pt_res1_out, "Residual1", i
        )

        keras_norm2_out = keras_block.norm2(keras_res1_out)
        pt_norm2_out = pt_block.norm2(pt_res1_out)
        keras_norm2_out = _compare_and_correct(
            keras_norm2_out, pt_norm2_out, "Norm2", i
        )

        keras_mlp_out = keras_block.mlp(keras_norm2_out, training=False)
        pt_mlp_out = pt_block.mlp(pt_norm2_out)
        keras_mlp_out = _compare_and_correct(keras_mlp_out, pt_mlp_out, "MLP", i)

        keras_ls2_out = keras_block.ls2(keras_mlp_out, training=False)
        pt_ls2_out = pt_block.ls2(pt_mlp_out)
        keras_ls2_out = _compare_and_correct(
            keras_ls2_out, pt_ls2_out, "LayerScale2", i
        )

        keras_final_out = keras_res1_out + keras_ls2_out
        pt_final_out = pt_res1_out + pt_ls2_out
        _compare_and_correct(keras_final_out, pt_final_out, "Final Block Output", i)

    print(f"\n--- Deep-Dive Analysis for {weight_key} Complete ---")
    assert (
        not any_failure
    ), f"One or more internal layers for {weight_key} had an output mismatch. See logs for details."


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main(["-v", __file__])
