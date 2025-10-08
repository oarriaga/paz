import os
import sys
import pytest
import torch
import numpy as np

os.environ["KERAS_BACKEND"] = "jax"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from paz.models.foundation.dinov3.layers import (
    Mlp,
    SwiGLUFFN,
)

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
# Test Suite
# ==============================================================================
def transfer_weights_from_pt_to_keras(pt_model, keras_model):
    pt_state_dict = pt_model.state_dict()

    # --- Standalone Weights ---
    keras_model.cls_token.assign(pt_state_dict["cls_token"].numpy())
    keras_model.mask_token.assign(pt_state_dict["mask_token"].numpy())
    if "storage_tokens" in pt_state_dict and keras_model.storage_tokens is not None:
        keras_model.storage_tokens.assign(pt_state_dict["storage_tokens"].numpy())

    # --- Patch Embedding ---
    pe_weights = [pt_state_dict["patch_embed.proj.weight"].permute(2, 3, 1, 0).numpy()]
    if "patch_embed.proj.bias" in pt_state_dict:
        pe_weights.append(pt_state_dict["patch_embed.proj.bias"].numpy())
    keras_model.patch_embed.projection.set_weights(pe_weights)

    # --- Transformer Blocks ---
    for i in range(len(keras_model.blocks)):
        block = keras_model.blocks[i]
        pt_block = pt_model.blocks[i]
        pt_block_prefix = f"blocks.{i}"

        # Norms
        norm1_weights = [pt_state_dict[f"{pt_block_prefix}.norm1.weight"].numpy()]
        if hasattr(pt_block.norm1, "bias") and pt_block.norm1.bias is not None:
            norm1_weights.append(pt_state_dict[f"{pt_block_prefix}.norm1.bias"].numpy())
        block.norm1.set_weights(norm1_weights)

        norm2_weights = [pt_state_dict[f"{pt_block_prefix}.norm2.weight"].numpy()]
        if hasattr(pt_block.norm2, "bias") and pt_block.norm2.bias is not None:
            norm2_weights.append(pt_state_dict[f"{pt_block_prefix}.norm2.bias"].numpy())
        block.norm2.set_weights(norm2_weights)

        # Attention Layers
        qkv_weights = [pt_state_dict[f"{pt_block_prefix}.attn.qkv.weight"].T.numpy()]
        if pt_block.attn.qkv.bias is not None:
            qkv_weights.append(
                pt_state_dict[f"{pt_block_prefix}.attn.qkv.bias"].numpy()
            )
        block.attn.qkv.set_weights(qkv_weights)

        proj_weights = [pt_state_dict[f"{pt_block_prefix}.attn.proj.weight"].T.numpy()]
        if pt_block.attn.proj.bias is not None:
            proj_weights.append(
                pt_state_dict[f"{pt_block_prefix}.attn.proj.bias"].numpy()
            )
        block.attn.proj.set_weights(proj_weights)

        # LayerScale
        if hasattr(block, "ls1") and hasattr(block.ls1, "gamma"):
            block.ls1.gamma.assign(
                pt_state_dict[f"{pt_block_prefix}.ls1.gamma"].numpy()
            )
        if hasattr(block, "ls2") and hasattr(block.ls2, "gamma"):
            block.ls2.gamma.assign(
                pt_state_dict[f"{pt_block_prefix}.ls2.gamma"].numpy()
            )

        # FFN Layers (Handles both Mlp and SwiGLUFFN)
        if isinstance(block.mlp, Mlp):
            fc1_weights = [pt_state_dict[f"{pt_block_prefix}.mlp.fc1.weight"].T.numpy()]
            if pt_block.mlp.fc1.bias is not None:
                fc1_weights.append(
                    pt_state_dict[f"{pt_block_prefix}.mlp.fc1.bias"].numpy()
                )
            block.mlp.fc1.set_weights(fc1_weights)
            fc2_weights = [pt_state_dict[f"{pt_block_prefix}.mlp.fc2.weight"].T.numpy()]
            if pt_block.mlp.fc2.bias is not None:
                fc2_weights.append(
                    pt_state_dict[f"{pt_block_prefix}.mlp.fc2.bias"].numpy()
                )
            block.mlp.fc2.set_weights(fc2_weights)
        elif isinstance(block.mlp, SwiGLUFFN):
            w1_weights = [pt_state_dict[f"{pt_block_prefix}.mlp.w1.weight"].T.numpy()]
            if pt_block.mlp.w1.bias is not None:
                w1_weights.append(
                    pt_state_dict[f"{pt_block_prefix}.mlp.w1.bias"].numpy()
                )
            block.mlp.w1.set_weights(w1_weights)
            w2_weights = [pt_state_dict[f"{pt_block_prefix}.mlp.w2.weight"].T.numpy()]
            if pt_block.mlp.w2.bias is not None:
                w2_weights.append(
                    pt_state_dict[f"{pt_block_prefix}.mlp.w2.bias"].numpy()
                )
            block.mlp.w2.set_weights(w2_weights)
            w3_weights = [pt_state_dict[f"{pt_block_prefix}.mlp.w3.weight"].T.numpy()]
            if pt_block.mlp.w3.bias is not None:
                w3_weights.append(
                    pt_state_dict[f"{pt_block_prefix}.mlp.w3.bias"].numpy()
                )
            block.mlp.w3.set_weights(w3_weights)

    # --- Final Layer Norm ---
    final_norm_weights = [pt_state_dict["norm.weight"].numpy()]
    if hasattr(keras_model.norm, "beta"):
        final_norm_weights.append(pt_state_dict["norm.bias"].numpy())
    keras_model.norm.set_weights(final_norm_weights)

    if "rope_embed.periods" in pt_state_dict:
        rope_periods = pt_state_dict["rope_embed.periods"].numpy()
        keras_model.rope_embed.set_weights([rope_periods])


# ==============================================================================
# Test Suite for small-Large (ViT-L/16)
# ==============================================================================


def test_final_output_equivalence_with_vitsmall():
    # --- 1. Define paths and model config ---
    DINO_REPO_PATH = (
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
    )
    DINO_WEIGHT_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    # Skip the test if the required files don't exist
    if not (os.path.isdir(DINO_REPO_PATH) and os.path.isfile(DINO_WEIGHT_PATH)):
        pytest.skip("DINOv3 repository or weight file not found. Skipping this test.")

    if DINO_REPO_PATH not in sys.path:
        sys.path.insert(0, DINO_REPO_PATH)

    # Configuration for dinov3_vits16.
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

    # --- 2. Create PyTorch and Keras Models using Factory Functions ---
    print("\nInstantiating models using vit_small and PT_vit_small...")

    pt_model = PT_vit_small(**model_kwargs)

    state_dict = torch.load(DINO_WEIGHT_PATH, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=False)

    pt_model.eval()
    print("PyTorch model created and pre-trained weights loaded.")

    keras_model = vit_small(**model_kwargs)
    dummy_input_np = np.random.randn(1, 224, 224, 3).astype("float32")
    keras_model(dummy_input_np)
    print("Keras model created and built.")

    # --- 3. Transfer Weights ---
    print("Transferring weights from PyTorch to Keras...")
    transfer_weights_from_pt_to_keras(pt_model, keras_model)
    print("Weight transfer complete.")

    # --- 4. Create Identical Inputs ---
    keras_input = keras.ops.convert_to_tensor(dummy_input_np)
    pt_input = torch.from_numpy(dummy_input_np.transpose(0, 3, 1, 2))
    print("Generated identical inputs for both frameworks.")

    # --- 5. Get Model Outputs ---
    print("Performing inference...")
    keras_output = keras_model(keras_input, training=False)
    with torch.no_grad():
        pt_output = pt_model(pt_input, is_training=False)

    # --- 6. Compare Outputs ---
    print("Comparing outputs...")
    keras_output_np = keras.ops.convert_to_numpy(keras_output)
    pt_output_np = pt_output.detach().cpu().numpy()
    try:
        np.testing.assert_allclose(
            keras_output_np,
            pt_output_np,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Final model outputs do not match after loading pre-trained weights",
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

    print(
        "\n✅ Success! Final outputs from vit_small and PT_vit_small match perfectly."
    )


def test_deep_dive_block_equivalence_vitsmall():
    # --- 1. Define paths and model config ---
    DINO_REPO_PATH = (
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
    )
    DINO_WEIGHT_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    if not (os.path.isdir(DINO_REPO_PATH) and os.path.isfile(DINO_WEIGHT_PATH)):
        pytest.skip("DINOv3 repository or weight file not found. Skipping this test.")

    if DINO_REPO_PATH not in sys.path:
        sys.path.insert(0, DINO_REPO_PATH)

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

    # --- 2. Create, Load, and Build Models ---
    print("\n--- Starting Deep-Dive Block Equivalence Test ---")
    print("Instantiating models using vit_small and PT_vit_small...")

    # Create and load the PyTorch model
    pt_model = PT_vit_small(**model_kwargs)
    state_dict = torch.load(DINO_WEIGHT_PATH, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=False)
    pt_model.eval()
    print("PyTorch model created and pre-trained weights loaded.")

    # Create and build the Keras model
    keras_model = vit_small(**model_kwargs)
    # Build the model by passing a dummy input so weights can be assigned
    keras_model(np.random.randn(1, 224, 224, 3).astype("float32"))
    print("Keras model created and built.")

    # --- 3. Transfer Weights ---
    print("Transferring weights from PyTorch to Keras...")
    transfer_weights_from_pt_to_keras(pt_model, keras_model)
    print("Weight transfer complete.")

    # --- 4. Deep Dive Comparison Logic ---
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

        # Layer 1: Normalization 1
        keras_norm1_out = keras_block.norm1(keras_tokens)
        pt_norm1_out = pt_block.norm1(pt_tokens)
        keras_norm1_out = _compare_and_correct(
            keras_norm1_out, pt_norm1_out, "Norm1", i
        )

        # Layer 2: Attention
        keras_attn_out = keras_block.attn(
            keras_norm1_out, rope=rope_keras, training=False
        )
        pt_attn_out = pt_block.attn(pt_norm1_out, rope=rope_torch)
        keras_attn_out = _compare_and_correct(
            keras_attn_out, pt_attn_out, "Attention", i
        )

        # Layer 3: LayerScale 1 & DropPath 1
        keras_ls1_out = keras_block.ls1(keras_attn_out, training=False)
        pt_ls1_out = pt_block.ls1(pt_attn_out)
        keras_ls1_out = _compare_and_correct(
            keras_ls1_out, pt_ls1_out, "LayerScale1", i
        )

        # Residual Connection 1
        keras_res1_out = keras_tokens + keras_ls1_out
        pt_res1_out = pt_tokens + pt_ls1_out
        keras_res1_out = _compare_and_correct(
            keras_res1_out, pt_res1_out, "Residual1", i
        )

        # Layer 4: Normalization 2
        keras_norm2_out = keras_block.norm2(keras_res1_out)
        pt_norm2_out = pt_block.norm2(pt_res1_out)
        keras_norm2_out = _compare_and_correct(
            keras_norm2_out, pt_norm2_out, "Norm2", i
        )

        # Layer 5: MLP (Feed-Forward Network)
        keras_mlp_out = keras_block.mlp(keras_norm2_out, training=False)
        pt_mlp_out = pt_block.mlp(pt_norm2_out)
        keras_mlp_out = _compare_and_correct(keras_mlp_out, pt_mlp_out, "MLP", i)

        # Layer 6: LayerScale 2 & DropPath 2
        keras_ls2_out = keras_block.ls2(keras_mlp_out, training=False)
        pt_ls2_out = pt_block.ls2(pt_mlp_out)
        keras_ls2_out = _compare_and_correct(
            keras_ls2_out, pt_ls2_out, "LayerScale2", i
        )

        # Residual Connection 2 (Final Block Output)
        keras_final_out = keras_res1_out + keras_ls2_out
        pt_final_out = pt_res1_out + pt_ls2_out
        _compare_and_correct(keras_final_out, pt_final_out, "Final Block Output", i)

    print("\n--- Deep-Dive Analysis Complete ---")
    assert (
        not any_failure
    ), "One or more internal layers had an output mismatch. See logs for details."


# ==============================================================================
# Test Suite for ViT-Base (ViT-B/16)
# ==============================================================================


def test_final_output_equivalence_with_vitbase():
    # --- 1. Define paths and model config ---
    DINO_REPO_PATH = (
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
    )
    DINO_WEIGHT_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    # Skip the test if the required files don't exist
    if not (os.path.isdir(DINO_REPO_PATH) and os.path.isfile(DINO_WEIGHT_PATH)):
        pytest.skip("DINOv3 repository or weight file not found. Skipping this test.")

    # Add DINO repo to Python path to allow torch.hub.load to find the model definition
    if DINO_REPO_PATH not in sys.path:
        sys.path.insert(0, DINO_REPO_PATH)

    # Configuration for dinov3_vitb16.
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

    # --- 2. Create PyTorch and Keras Models using Factory Functions ---
    print("\nInstantiating models using vit_base and PT_vit_base...")

    # Create the PyTorch model using its factory function
    pt_model = PT_vit_base(**model_kwargs)

    # Load the pre-trained weights into the instantiated model
    state_dict = torch.load(DINO_WEIGHT_PATH, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=False)

    pt_model.eval()
    print("PyTorch model created and pre-trained weights loaded.")

    # Create the Keras model using its factory function
    keras_model = vit_base(**model_kwargs)
    # Build the Keras model by passing a dummy input
    dummy_input_np = np.random.randn(1, 224, 224, 3).astype("float32")
    keras_model(dummy_input_np)
    print("Keras model created and built.")

    # --- 3. Transfer Weights ---
    print("Transferring weights from PyTorch to Keras...")
    transfer_weights_from_pt_to_keras(pt_model, keras_model)
    print("Weight transfer complete.")

    # --- 4. Create Identical Inputs ---
    keras_input = keras.ops.convert_to_tensor(dummy_input_np)
    pt_input = torch.from_numpy(dummy_input_np.transpose(0, 3, 1, 2))
    print("Generated identical inputs for both frameworks.")

    # --- 5. Get Model Outputs ---
    print("Performing inference...")
    # Get Keras output (inference mode)
    keras_output = keras_model(keras_input, training=False)
    # Get PyTorch output (inference mode)
    with torch.no_grad():
        pt_output = pt_model(pt_input, is_training=False)

    # --- 6. Compare Outputs ---
    print("Comparing outputs...")
    keras_output_np = keras.ops.convert_to_numpy(keras_output)
    pt_output_np = pt_output.detach().cpu().numpy()

    # Assert that the outputs are numerically close within a tolerance
    try:
        np.testing.assert_allclose(
            keras_output_np,
            pt_output_np,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Final model outputs do not match after loading pre-trained weights",
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

    print("\n✅ Success! Final outputs from vit_base and PT_vit_base match perfectly.")


def test_deep_dive_block_equivalence_vitbase():
    # --- 1. Define paths and model config ---
    DINO_REPO_PATH = (
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
    )
    DINO_WEIGHT_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    if not (os.path.isdir(DINO_REPO_PATH) and os.path.isfile(DINO_WEIGHT_PATH)):
        pytest.skip("DINOv3 repository or weight file not found. Skipping this test.")

    if DINO_REPO_PATH not in sys.path:
        sys.path.insert(0, DINO_REPO_PATH)

    # Configuration for dinov3_vitb16.
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

    # --- 2. Create, Load, and Build Models ---
    print("\n--- Starting Deep-Dive Block Equivalence Test for ViT-Base ---")
    print("Instantiating models using vit_base and PT_vit_base...")

    # Create and load the PyTorch model
    pt_model = PT_vit_base(**model_kwargs)
    state_dict = torch.load(DINO_WEIGHT_PATH, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=False)
    pt_model.eval()
    print("PyTorch model created and pre-trained weights loaded.")

    # Create and build the Keras model
    keras_model = vit_base(**model_kwargs)
    # Build the model by passing a dummy input so weights can be assigned
    keras_model(np.random.randn(1, 224, 224, 3).astype("float32"))
    print("Keras model created and built.")

    # --- 3. Transfer Weights ---
    print("Transferring weights from PyTorch to Keras...")
    transfer_weights_from_pt_to_keras(pt_model, keras_model)
    print("Weight transfer complete.")

    # --- 4. Deep Dive Comparison Logic  ---
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
            if mean_diff > 1e-4:  # Using a stricter tolerance for deep dive
                raise AssertionError(
                    f"Mean absolute difference {mean_diff} exceeds tolerance of 1e-4."
                )
            print(f"    ✅ Block {block_idx} - {layer_name}: Output matches.")
            return keras_tensor  # Return the original Keras tensor if it matches
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

        # Layer 1: Normalization 1
        keras_norm1_out = keras_block.norm1(keras_tokens)
        pt_norm1_out = pt_block.norm1(pt_tokens)
        keras_norm1_out = _compare_and_correct(
            keras_norm1_out, pt_norm1_out, "Norm1", i
        )

        # Layer 2: Attention
        keras_attn_out = keras_block.attn(
            keras_norm1_out, rope=rope_keras, training=False
        )
        pt_attn_out = pt_block.attn(pt_norm1_out, rope=rope_torch)
        keras_attn_out = _compare_and_correct(
            keras_attn_out, pt_attn_out, "Attention", i
        )

        # Layer 3: LayerScale 1 & DropPath 1 (DropPath is identity in eval mode)
        keras_ls1_out = keras_block.ls1(keras_attn_out, training=False)
        pt_ls1_out = pt_block.ls1(pt_attn_out)
        keras_ls1_out = _compare_and_correct(
            keras_ls1_out, pt_ls1_out, "LayerScale1", i
        )

        # Residual Connection 1
        keras_res1_out = keras_tokens + keras_ls1_out
        pt_res1_out = pt_tokens + pt_ls1_out
        keras_res1_out = _compare_and_correct(
            keras_res1_out, pt_res1_out, "Residual1", i
        )

        # Layer 4: Normalization 2
        keras_norm2_out = keras_block.norm2(keras_res1_out)
        pt_norm2_out = pt_block.norm2(pt_res1_out)
        keras_norm2_out = _compare_and_correct(
            keras_norm2_out, pt_norm2_out, "Norm2", i
        )

        # Layer 5: MLP (Feed-Forward Network)
        keras_mlp_out = keras_block.mlp(keras_norm2_out, training=False)
        pt_mlp_out = pt_block.mlp(pt_norm2_out)
        keras_mlp_out = _compare_and_correct(keras_mlp_out, pt_mlp_out, "MLP", i)

        # Layer 6: LayerScale 2 & DropPath 2 (DropPath is identity in eval mode)
        keras_ls2_out = keras_block.ls2(keras_mlp_out, training=False)
        pt_ls2_out = pt_block.ls2(pt_mlp_out)
        keras_ls2_out = _compare_and_correct(
            keras_ls2_out, pt_ls2_out, "LayerScale2", i
        )

        # Residual Connection 2 (Final Block Output)
        keras_final_out = keras_res1_out + keras_ls2_out
        pt_final_out = pt_res1_out + pt_ls2_out
        _compare_and_correct(keras_final_out, pt_final_out, "Final Block Output", i)

    print("\n--- Deep-Dive Analysis for ViT-Base Complete ---")
    assert (
        not any_failure
    ), "One or more internal layers for ViT-Base had an output mismatch. See logs for details."


# ==============================================================================
# Test Suite for ViT-Large (ViT-L/16)
# ==============================================================================


def test_final_output_equivalence_with_vitlarge():
    # --- 1. Define paths and model config ---
    DINO_REPO_PATH = (
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
    )
    DINO_WEIGHT_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

    if not (os.path.isdir(DINO_REPO_PATH) and os.path.isfile(DINO_WEIGHT_PATH)):
        pytest.skip("DINOv3 repository or weight file not found. Skipping this test.")

    if DINO_REPO_PATH not in sys.path:
        sys.path.insert(0, DINO_REPO_PATH)

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

    # --- 2. Create PyTorch and Keras Models using Factory Functions ---
    print("\nInstantiating models using vit_large and PT_vit_large...")

    pt_model = PT_vit_large(**model_kwargs)

    # Load the pre-trained weights into the instantiated model
    state_dict = torch.load(DINO_WEIGHT_PATH, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=False)

    pt_model.eval()
    print("PyTorch model created and pre-trained weights loaded.")

    # Create the Keras model using its factory function
    keras_model = vit_large(**model_kwargs)
    # Build the Keras model by passing a dummy input
    dummy_input_np = np.random.randn(1, 224, 224, 3).astype("float32")
    keras_model(dummy_input_np)
    print("Keras model created and built.")

    # --- 3. Transfer Weights ---
    print("Transferring weights from PyTorch to Keras...")
    transfer_weights_from_pt_to_keras(pt_model, keras_model)
    print("Weight transfer complete.")

    # --- 4. Create Identical Inputs ---
    keras_input = keras.ops.convert_to_tensor(dummy_input_np)
    pt_input = torch.from_numpy(dummy_input_np.transpose(0, 3, 1, 2))
    print("Generated identical inputs for both frameworks.")

    # --- 5. Get Model Outputs ---
    print("Performing inference...")
    # Get Keras output (inference mode)
    keras_output = keras_model(keras_input, training=False)
    # Get PyTorch output (inference mode)
    with torch.no_grad():
        pt_output = pt_model(pt_input, is_training=False)

    # --- 6. Compare Outputs ---
    print("Comparing outputs...")
    # Convert both outputs to numpy arrays for comparison
    keras_output_np = keras.ops.convert_to_numpy(keras_output)
    pt_output_np = pt_output.detach().cpu().numpy()

    # Assert that the outputs are numerically close within a tolerance
    try:
        np.testing.assert_allclose(
            keras_output_np,
            pt_output_np,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Final model outputs do not match after loading pre-trained weights",
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

    print(
        "\n✅ Success! Final outputs from vit_large and PT_vit_large match perfectly."
    )


def test_deep_dive_block_equivalence_vitlarge():
    # --- 1. Define paths and model config ---
    DINO_REPO_PATH = (
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
    )
    DINO_WEIGHT_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

    # Skip the test if the required files don't exist
    if not (os.path.isdir(DINO_REPO_PATH) and os.path.isfile(DINO_WEIGHT_PATH)):
        pytest.skip("DINOv3 repository or weight file not found. Skipping this test.")

    # Add DINO repo to Python path to allow torch.hub.load to find the model definition
    if DINO_REPO_PATH not in sys.path:
        sys.path.insert(0, DINO_REPO_PATH)

    # Configuration for dinov3_vitl16.
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

    # --- 2. Create, Load, and Build Models ---
    print("\n--- Starting Deep-Dive Block Equivalence Test for ViT-Large ---")
    print("Instantiating models using vit_large and PT_vit_large...")

    # Create and load the PyTorch model
    pt_model = PT_vit_large(**model_kwargs)
    state_dict = torch.load(DINO_WEIGHT_PATH, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=False)
    pt_model.eval()
    print("PyTorch model created and pre-trained weights loaded.")

    # Create and build the Keras model
    keras_model = vit_large(**model_kwargs)
    # Build the model by passing a dummy input so weights can be assigned
    keras_model(np.random.randn(1, 224, 224, 3).astype("float32"))
    print("Keras model created and built.")

    # --- 3. Transfer Weights ---
    print("Transferring weights from PyTorch to Keras...")
    transfer_weights_from_pt_to_keras(pt_model, keras_model)
    print("Weight transfer complete.")

    # --- 4. Deep Dive Comparison Logic ---
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

    # Sequentially test each block
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

        # Layer 1: Normalization 1
        keras_norm1_out = keras_block.norm1(keras_tokens)
        pt_norm1_out = pt_block.norm1(pt_tokens)
        keras_norm1_out = _compare_and_correct(
            keras_norm1_out, pt_norm1_out, "Norm1", i
        )

        # Layer 2: Attention
        keras_attn_out = keras_block.attn(
            keras_norm1_out, rope=rope_keras, training=False
        )
        pt_attn_out = pt_block.attn(pt_norm1_out, rope=rope_torch)
        keras_attn_out = _compare_and_correct(
            keras_attn_out, pt_attn_out, "Attention", i
        )

        # Layer 3: LayerScale 1 & DropPath 1 (DropPath is identity in eval mode)
        keras_ls1_out = keras_block.ls1(keras_attn_out, training=False)
        pt_ls1_out = pt_block.ls1(pt_attn_out)
        keras_ls1_out = _compare_and_correct(
            keras_ls1_out, pt_ls1_out, "LayerScale1", i
        )

        # Residual Connection 1
        keras_res1_out = keras_tokens + keras_ls1_out
        pt_res1_out = pt_tokens + pt_ls1_out
        keras_res1_out = _compare_and_correct(
            keras_res1_out, pt_res1_out, "Residual1", i
        )

        # Layer 4: Normalization 2
        keras_norm2_out = keras_block.norm2(keras_res1_out)
        pt_norm2_out = pt_block.norm2(pt_res1_out)
        keras_norm2_out = _compare_and_correct(
            keras_norm2_out, pt_norm2_out, "Norm2", i
        )

        # Layer 5: MLP (Feed-Forward Network)
        keras_mlp_out = keras_block.mlp(keras_norm2_out, training=False)
        pt_mlp_out = pt_block.mlp(pt_norm2_out)
        keras_mlp_out = _compare_and_correct(keras_mlp_out, pt_mlp_out, "MLP", i)

        # Layer 6: LayerScale 2 & DropPath 2
        keras_ls2_out = keras_block.ls2(keras_mlp_out, training=False)
        pt_ls2_out = pt_block.ls2(pt_mlp_out)
        keras_ls2_out = _compare_and_correct(
            keras_ls2_out, pt_ls2_out, "LayerScale2", i
        )

        # Residual Connection 2 (Final Block Output)
        keras_final_out = keras_res1_out + keras_ls2_out
        pt_final_out = pt_res1_out + pt_ls2_out
        _compare_and_correct(keras_final_out, pt_final_out, "Final Block Output", i)

    print("\n--- Deep-Dive Analysis for ViT-Large Complete ---")
    assert (
        not any_failure
    ), "One or more internal layers for ViT-Large had an output mismatch. See logs for details."


if __name__ == "__main__":
    pytest.main(["-v", __file__])
