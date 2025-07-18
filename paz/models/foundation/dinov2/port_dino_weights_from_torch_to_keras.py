import os

os.environ["KERAS_BACKEND"] = "jax"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))

import sys

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import keras
import torch
import numpy as np
from paz.models.foundation.dinov2.models.vision_transformer import (
    vit_small,
    vit_large,
    vit_base,
    vit_giant2,
    PatchEmbed,
)
import torch.nn.functional
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def port_weights_from_state_dict(keras_model, state_dict):
    """
    Loads weights from a PyTorch state_dict into a native Keras DINOv2 model.

    This version is corrected to handle potential LayerScale (`ls`) parameters,
    dynamically determines the number of blocks, and accepts a state_dict
    directly instead of a file path.

    Args:
        keras_model: An instance of a Keras DINOv2 ViT model.
        state_dict: The state_dict from the pretrained PyTorch model.
    """
    logging.info(f"Starting weight porting for Keras model '{keras_model.name}'...")

    keras_model.classification_token.assign(state_dict["cls_token"].numpy())
    logging.info("✓ Ported classification_token.")

    torch_positional_embedding = state_dict["pos_embed"]
    keras_positional_embedding_shape = keras_model.positional_embedding.shape

    if torch_positional_embedding.shape != keras_positional_embedding_shape:
        logging.info(
            f"  - Resizing positional embedding from {list(torch_positional_embedding.shape)} to {list(keras_positional_embedding_shape)}..."
        )
        torch_cls_position_embedding = torch_positional_embedding[:, :1, :]
        torch_patch_position_embedding = torch_positional_embedding[:, 1:, :]
        number_of_source_patches = torch_patch_position_embedding.shape[1]
        number_of_target_patches = keras_positional_embedding_shape[1] - 1
        source_grid_size = int(np.sqrt(number_of_source_patches))
        target_grid_size = int(np.sqrt(number_of_target_patches))
        embedding_dimension = keras_positional_embedding_shape[2]
        torch_patch_position_embedding_grid = torch_patch_position_embedding.reshape(
            1, source_grid_size, source_grid_size, embedding_dimension
        ).permute(0, 3, 1, 2)
        resized_patch_position_embedding_grid = torch.nn.functional.interpolate(
            torch_patch_position_embedding_grid,
            size=(target_grid_size, target_grid_size),
            mode="bicubic",
            align_corners=False,
        )
        resized_patch_position_embedding = (
            resized_patch_position_embedding_grid.permute(0, 2, 3, 1).reshape(
                1, -1, embedding_dimension
            )
        )
        final_positional_embedding = torch.cat(
            [torch_cls_position_embedding, resized_patch_position_embedding], dim=1
        )
        keras_model.positional_embedding.assign(final_positional_embedding.numpy())
    else:
        keras_model.positional_embedding.assign(torch_positional_embedding.numpy())
    logging.info("✓ Ported positional_embedding.")

    patch_embedding_layer = None
    for layer in keras_model.layers:
        if isinstance(layer, PatchEmbed):
            patch_embedding_layer = layer
            logging.info(
                f"Found PatchEmbed layer by its class: '{patch_embedding_layer.name}'"
            )
            break

    if patch_embedding_layer:
        patch_embedding_w = (
            state_dict["patch_embed.proj.weight"].permute(2, 3, 1, 0).numpy()
        )
        patch_embedding_b = state_dict["patch_embed.proj.bias"].numpy()
        patch_embedding_layer.projection_layer.set_weights(
            [patch_embedding_w, patch_embedding_b]
        )
        logging.info("✓ Ported patch_embed weights.")
    else:
        raise RuntimeError("Could not find the PatchEmbed layer in the Keras model.")

    chunk_layer = keras_model.get_layer("chunk_0")
    number_of_blocks = len(chunk_layer.blocks)
    logging.info(f"Found {number_of_blocks} transformer blocks to port...")

    for i in range(number_of_blocks):
        block = chunk_layer.blocks[i]
        block.normalization1.set_weights(
            [
                state_dict[f"blocks.{i}.norm1.weight"].numpy(),
                state_dict[f"blocks.{i}.norm1.bias"].numpy(),
            ]
        )
        block.normalization2.set_weights(
            [
                state_dict[f"blocks.{i}.norm2.weight"].numpy(),
                state_dict[f"blocks.{i}.norm2.bias"].numpy(),
            ]
        )
        block.attention.predict_query_key_value.set_weights(
            [
                state_dict[f"blocks.{i}.attn.qkv.weight"].T.numpy(),
                state_dict[f"blocks.{i}.attn.qkv.bias"].numpy(),
            ]
        )
        block.attention.projection_layer.set_weights(
            [
                state_dict[f"blocks.{i}.attn.proj.weight"].T.numpy(),
                state_dict[f"blocks.{i}.attn.proj.bias"].numpy(),
            ]
        )

        mlp_fully_connected_layer_1_key = f"blocks.{i}.mlp.fc1.weight"
        if mlp_fully_connected_layer_1_key in state_dict:
            block.mlp.fully_connected_layer_1.set_weights(
                [
                    state_dict[f"blocks.{i}.mlp.fc1.weight"].T.numpy(),
                    state_dict[f"blocks.{i}.mlp.fc1.bias"].numpy(),
                ]
            )
            block.mlp.fully_connected_layer_2.set_weights(
                [
                    state_dict[f"blocks.{i}.mlp.fc2.weight"].T.numpy(),
                    state_dict[f"blocks.{i}.mlp.fc2.bias"].numpy(),
                ]
            )
        else:
            # Check for SwiGLU weights, using the new descriptive names for the Keras model
            swiglu_w12_key = f"blocks.{i}.mlp.w12.weight"
            swiglu_w3_key = f"blocks.{i}.mlp.w3.weight"
            if swiglu_w12_key in state_dict:
                # Porting to Keras layers with new descriptive names
                block.mlp.fused_gate_and_value_projection.set_weights(
                    [
                        state_dict[swiglu_w12_key].T.numpy(),
                        state_dict[f"blocks.{i}.mlp.w12.bias"].numpy(),
                    ]
                )
                block.mlp.output_projection.set_weights(
                    [
                        state_dict[swiglu_w3_key].T.numpy(),
                        state_dict[f"blocks.{i}.mlp.w3.bias"].numpy(),
                    ]
                )
            else:
                raise KeyError(
                    f"Could not find FFN weights for block {i}. "
                    "Neither standard MLP nor SwiGLU keys were found."
                )
        # LayerScale parameters
        layer_scale_1_key = f"blocks.{i}.ls1.gamma"
        layer_scale_2_key = f"blocks.{i}.ls2.gamma"
        if layer_scale_1_key in state_dict and hasattr(block, "layer_scale_1"):
            block.layer_scale_1.gamma.assign(state_dict[layer_scale_1_key].numpy())
        if layer_scale_2_key in state_dict and hasattr(block, "layer_scale_2"):
            block.layer_scale_2.gamma.assign(state_dict[layer_scale_2_key].numpy())

    logging.info("✓ Ported all transformer blocks.")

    # --- Port final Layer Normalization ---
    final_normalization_layer = keras_model.get_layer("norm")
    if final_normalization_layer:
        final_normalization_layer.set_weights(
            [state_dict["norm.weight"].numpy(), state_dict["norm.bias"].numpy()]
        )
        logging.info("✓ Ported final LayerNormalization.")

    logging.info("\n--- Weight porting complete! ---")
    return keras_model


if __name__ == "__main__":
    MODEL_CONSTRUCTORS = {
        "dinov2_vits14": vit_small,
        "dinov2_vitb14": vit_base,
        "dinov2_vitl14": vit_large,
        # "dinov2_vitg14": vit_giant2
    }

    output_dir = "weights"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set to: '{output_dir}'")

    for model_name, model_constructor in MODEL_CONSTRUCTORS.items():
        print("-" * 80)
        logging.info(f"Processing model: {model_name}")
        logging.info(f"Using Keras backend: {keras.backend.backend()}")

        logging.info("Loading pretrained DINOv2 model from Torch Hub...")
        pytorch_model = torch.hub.load(
            "facebookresearch/dinov2", model_name, verbose=False
        )
        pytorch_model.eval()
        pytorch_state_dict = pytorch_model.state_dict()
        logging.info("✓ PyTorch model and weights loaded successfully.")

        logging.info("Instantiating Keras 3 DINOv2 model architecture...")
        keras_dinov2_model = model_constructor(
            img_size=518,
            patch_size=14,
            init_values=1.0e-5,
            number_of_register_tokens=0,
            name=f"dino_{model_name}",
        )

        logging.info("Building Keras model to initialize weights...")
        dummy_input = np.zeros((1, 518, 518, 3), dtype="float32")
        _ = keras_dinov2_model(dummy_input)

        keras_dinov2_model_with_weights = port_weights_from_state_dict(
            keras_dinov2_model, pytorch_state_dict
        )

        final_keras_model_path = os.path.join(output_dir, f"{model_name}_ported.keras")
        logging.info(
            f"Saving final native Keras model to '{final_keras_model_path}'..."
        )
        keras_dinov2_model_with_weights.save(final_keras_model_path)
        logging.info("✓ Save complete.")
        print("-" * 80 + "\n")

    print("All models have been successfully ported and saved.")
