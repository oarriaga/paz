import os
import sys
import logging
import torch
import numpy as np

# --- Configuration for Keras Backend and Environment ---

os.environ["KERAS_BACKEND"] = "jax"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

# ==============================================================================
# Keras Layer Implementation
# ==============================================================================
from paz.models.foundation.dinov3.models.vision_transformer import (
    vit_small,
    vit_base,
    vit_large,
)

# ==============================================================================
# PyTorch to Keras Weight Porting Function (Adapted for DINOv3)
# ==============================================================================


def transfer_weights_from_pt_to_keras(pt_state_dict, keras_model):
    """
    Loads weights from a PyTorch DINOv3 state_dict into a native Keras DINOv3 model.

    Args:
        pt_state_dict: The state_dict from the pretrained PyTorch model.
        keras_model: An instance of a Keras DinoVisionTransformer model.
    """
    logging.info(f"Starting weight porting for Keras model '{keras_model.name}'...")

    # --- Standalone Weights (cls_token, mask_token, storage_tokens) ---
    keras_model.cls_token.assign(pt_state_dict["cls_token"].numpy())
    logging.info("✓ Ported cls_token.")

    keras_model.mask_token.assign(pt_state_dict["mask_token"].numpy())
    logging.info("✓ Ported mask_token.")

    if "storage_tokens" in pt_state_dict and keras_model.storage_tokens is not None:
        keras_model.storage_tokens.assign(pt_state_dict["storage_tokens"].numpy())
        logging.info("✓ Ported storage_tokens.")

    # --- Patch Embedding ---
    patch_embed_layer = keras_model.patch_embed

    pe_weights = [pt_state_dict["patch_embed.proj.weight"].permute(2, 3, 1, 0).numpy()]
    if "patch_embed.proj.bias" in pt_state_dict:
        pe_weights.append(pt_state_dict["patch_embed.proj.bias"].numpy())

    patch_embed_layer.projection.set_weights(pe_weights)
    logging.info("✓ Ported patch_embed weights.")

    # --- RoPE Embedding (Only periods/inv_freq is set if present) ---
    if "rope_embed.periods" in pt_state_dict:
        rope_periods = pt_state_dict["rope_embed.periods"].float().numpy()
        keras_model.rope_embed.set_weights([rope_periods])
        logging.info("✓ Ported rope_embed periods.")

    # --- Transformer Blocks ---
    number_of_blocks = len(keras_model.blocks)
    logging.info(f"Found {number_of_blocks} transformer blocks to port...")

    for i in range(number_of_blocks):
        block = keras_model.blocks[i]
        pt_block_prefix = f"blocks.{i}"

        # Norms (StableLayerNormalization/RMSNorm inherit set_weights from Layer)
        block.norm1.set_weights(
            [
                pt_state_dict[f"{pt_block_prefix}.norm1.weight"].numpy(),
                pt_state_dict[f"{pt_block_prefix}.norm1.bias"].numpy(),
            ]
        )
        block.norm2.set_weights(
            [
                pt_state_dict[f"{pt_block_prefix}.norm2.weight"].numpy(),
                pt_state_dict[f"{pt_block_prefix}.norm2.bias"].numpy(),
            ]
        )

        # Attention Layers
        block.attn.qkv.set_weights(
            [
                pt_state_dict[f"{pt_block_prefix}.attn.qkv.weight"].T.numpy(),
                pt_state_dict[f"{pt_block_prefix}.attn.qkv.bias"].numpy(),
            ]
        )
        block.attn.proj.set_weights(
            [
                pt_state_dict[f"{pt_block_prefix}.attn.proj.weight"].T.numpy(),
                pt_state_dict[f"{pt_block_prefix}.attn.proj.bias"].numpy(),
            ]
        )

        # LayerScale (ls1 and ls2)
        layer_scale_1_key = f"{pt_block_prefix}.ls1.gamma"
        if layer_scale_1_key in pt_state_dict and hasattr(block, "ls1"):
            block.ls1.gamma.assign(pt_state_dict[layer_scale_1_key].numpy())

        layer_scale_2_key = f"{pt_block_prefix}.ls2.gamma"
        if layer_scale_2_key in pt_state_dict and hasattr(block, "ls2"):
            block.ls2.gamma.assign(pt_state_dict[layer_scale_2_key].numpy())

        # FFN Layers (Handles both Mlp and SwiGLUFFN)
        # Check FFN type by checking keys for Mlp first
        if f"{pt_block_prefix}.mlp.fc1.weight" in pt_state_dict:
            # Mlp (fc1, fc2)
            block.mlp.fc1.set_weights(
                [
                    pt_state_dict[f"{pt_block_prefix}.mlp.fc1.weight"].T.numpy(),
                    pt_state_dict[f"{pt_block_prefix}.mlp.fc1.bias"].numpy(),
                ]
            )
            block.mlp.fc2.set_weights(
                [
                    pt_state_dict[f"{pt_block_prefix}.mlp.fc2.weight"].T.numpy(),
                    pt_state_dict[f"{pt_block_prefix}.mlp.fc2.bias"].numpy(),
                ]
            )
        elif f"{pt_block_prefix}.mlp.w1.weight" in pt_state_dict:
            # SwiGLUFFN (w1, w2, w3) - as found in vision_transformer_test.py
            block.mlp.w1.set_weights(
                [
                    pt_state_dict[f"{pt_block_prefix}.mlp.w1.weight"].T.numpy(),
                    pt_state_dict[f"{pt_block_prefix}.mlp.w1.bias"].numpy(),
                ]
            )
            block.mlp.w2.set_weights(
                [
                    pt_state_dict[f"{pt_block_prefix}.mlp.w2.weight"].T.numpy(),
                    pt_state_dict[f"{pt_block_prefix}.mlp.w2.bias"].numpy(),
                ]
            )
            block.mlp.w3.set_weights(
                [
                    pt_state_dict[f"{pt_block_prefix}.mlp.w3.weight"].T.numpy(),
                    pt_state_dict[f"{pt_block_prefix}.mlp.w3.bias"].numpy(),
                ]
            )
        else:
            raise KeyError(
                f"Could not find FFN weights for block {i}. "
                "Neither standard MLP nor SwiGLUFFN keys were found."
            )

        logging.info(f"✓ Ported block {i}.")

    # --- Final Normalization Layers ---
    keras_model.norm.set_weights(
        [
            pt_state_dict["norm.weight"].numpy(),
            pt_state_dict["norm.bias"].numpy(),
        ]
    )
    logging.info("✓ Ported final 'norm' layer.")

    if keras_model.cls_norm:
        keras_model.cls_norm.set_weights(
            [
                pt_state_dict["cls_norm.weight"].numpy(),
                pt_state_dict["cls_norm.bias"].numpy(),
            ]
        )
        logging.info("✓ Ported final 'cls_norm' layer.")

    if keras_model.local_cls_norm:
        keras_model.local_cls_norm.set_weights(
            [
                pt_state_dict["local_cls_norm.weight"].numpy(),
                pt_state_dict["local_cls_norm.bias"].numpy(),
            ]
        )
        logging.info("✓ Ported final 'local_cls_norm' layer.")

    logging.info("\n--- Weight porting complete! ---")
    return keras_model


# ==============================================================================
# Execution Script
# ==============================================================================

if __name__ == "__main__":

    DINO_REPO_PATH = (
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
    )
    MODELS_WEIGHTS_DIR_PATH = (
        r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3/"
    )
    MODEL_CONSTRUCTORS = {
        "dinov3_vits16": vit_small,
        "dinov3_vitb16": vit_base,
        "dinov3_vitl16": vit_large,
    }

    MODEL_WEIGHTS_PATHS = {
        "dinov3_vits16": r"dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vitb16": r"dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": r"dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
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

    output_dir = "weights_dinov3"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set to: '{output_dir}'")

    MODEL_CONFIGS = {}
    for name, constructor in MODEL_CONSTRUCTORS.items():
        MODEL_CONFIGS[name] = {
            "constructor": constructor,
            "weight_filename": MODEL_WEIGHTS_PATHS[name],
            "keras_name": f"dino_{name}",
            "kwargs": model_kwargs,
        }

    for model_key, config in MODEL_CONFIGS.items():
        print("-" * 80)
        logging.info(f"Processing model: {model_key}")

        # --- Load PyTorch Weights ---
        # Safely join the weights directory path and the specific filename
        weight_path = os.path.join(MODELS_WEIGHTS_DIR_PATH, config["weight_filename"])

        if not os.path.isfile(weight_path):
            logging.error(f"PyTorch weight file not found at: {weight_path}. Skipping.")
            continue

        logging.info(f"Loading PyTorch state dict from: {weight_path}...")
        # Map to CPU to avoid GPU memory issues if unnecessary
        pytorch_state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
        logging.info("✓ PyTorch weights loaded successfully.")

        # --- Instantiate Keras Model ---
        logging.info("Instantiating Keras 3 DINOv3 model architecture...")
        keras_dinov3_model = config["constructor"](
            name=config["keras_name"], **config["kwargs"]
        )

        # --- Build Keras Model ---
        logging.info("Building Keras model to initialize weights...")
        img_size = config["kwargs"]["img_size"]
        dummy_input = np.zeros((1, img_size, img_size, 3), dtype="float32")
        # Call the model in inference mode to ensure all layers are built
        _ = keras_dinov3_model(dummy_input, training=False)
        logging.info("✓ Keras model built.")

        # --- Port Weights ---
        keras_dinov3_model_with_weights = transfer_weights_from_pt_to_keras(
            pytorch_state_dict, keras_dinov3_model
        )

        # --- Save Keras Model ---
        final_keras_model_path = os.path.join(output_dir, f"{model_key}_ported.keras")
        logging.info(
            f"Saving final native Keras model to '{final_keras_model_path}'..."
        )
        keras_dinov3_model_with_weights.save(final_keras_model_path)
        logging.info("✓ Save complete.")
        print("-" * 80 + "\n")

    print("All configured models have been successfully ported and saved.")
