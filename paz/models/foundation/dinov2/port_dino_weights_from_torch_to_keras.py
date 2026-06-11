import os

os.environ["KERAS_BACKEND"] = "jax"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import keras
import torch
import numpy as np
from paz.models.foundation.dinov2.models.vision_transformer import (
    DINOV2Small,
    DINOV2Base,
    DINOV2Large,
    DINOV2Giant2,
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

    cls_token_layer = keras_model.get_layer("cls_token")
    embedding_dimension = cls_token_layer.embeddings.shape[-1]
    cls_torch = state_dict["cls_token"].numpy().reshape(1, embedding_dimension)
    cls_token_layer.embeddings.assign(cls_torch)
    logging.info("✓ Ported classification_token.")

    pos_embed_layer = keras_model.get_layer("pos_embed")
    torch_positional_embedding = state_dict["pos_embed"]
    target_length = pos_embed_layer.embeddings.shape[0]
    target_shape = (1, target_length, embedding_dimension)

    if tuple(torch_positional_embedding.shape) != target_shape:
        logging.info(
            f"  - Resizing positional embedding from {list(torch_positional_embedding.shape)} to {list(target_shape)}..."
        )
        torch_cls_position_embedding = torch_positional_embedding[:, :1, :]
        torch_patch_position_embedding = torch_positional_embedding[:, 1:, :]
        number_of_source_patches = torch_patch_position_embedding.shape[1]
        number_of_target_patches = target_length - 1
        source_grid_size = int(np.sqrt(number_of_source_patches))
        target_grid_size = int(np.sqrt(number_of_target_patches))
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
        pos_embed_layer.embeddings.assign(
            final_positional_embedding.numpy().reshape(
                target_length, embedding_dimension
            )
        )
    else:
        pos_embed_layer.embeddings.assign(
            torch_positional_embedding.numpy().reshape(
                target_length, embedding_dimension
            )
        )
    logging.info("✓ Ported positional_embedding.")

    patch_embedding_w = (
        state_dict["patch_embed.proj.weight"].permute(2, 3, 1, 0).numpy()
    )
    patch_embedding_b = state_dict["patch_embed.proj.bias"].numpy()
    keras_model.get_layer("patch_embed_proj").set_weights(
        [patch_embedding_w, patch_embedding_b]
    )
    logging.info("✓ Ported patch_embed weights.")

    number_of_blocks = count_blocks(keras_model)
    logging.info(f"Found {number_of_blocks} transformer blocks to port...")

    for i in range(number_of_blocks):
        block_prefix = f"block_{i}"
        keras_model.get_layer(f"{block_prefix}_norm1").set_weights(
            [
                state_dict[f"blocks.{i}.norm1.weight"].numpy(),
                state_dict[f"blocks.{i}.norm1.bias"].numpy(),
            ]
        )
        keras_model.get_layer(f"{block_prefix}_norm2").set_weights(
            [
                state_dict[f"blocks.{i}.norm2.weight"].numpy(),
                state_dict[f"blocks.{i}.norm2.bias"].numpy(),
            ]
        )
        keras_model.get_layer(f"{block_prefix}_qkv").set_weights(
            [
                state_dict[f"blocks.{i}.attn.qkv.weight"].T.numpy(),
                state_dict[f"blocks.{i}.attn.qkv.bias"].numpy(),
            ]
        )
        keras_model.get_layer(f"{block_prefix}_proj").set_weights(
            [
                state_dict[f"blocks.{i}.attn.proj.weight"].T.numpy(),
                state_dict[f"blocks.{i}.attn.proj.bias"].numpy(),
            ]
        )

        mlp_fully_connected_layer_1_key = f"blocks.{i}.mlp.fc1.weight"
        if mlp_fully_connected_layer_1_key in state_dict:
            keras_model.get_layer(f"{block_prefix}_mlp_fc1").set_weights(
                [
                    state_dict[f"blocks.{i}.mlp.fc1.weight"].T.numpy(),
                    state_dict[f"blocks.{i}.mlp.fc1.bias"].numpy(),
                ]
            )
            keras_model.get_layer(f"{block_prefix}_mlp_fc2").set_weights(
                [
                    state_dict[f"blocks.{i}.mlp.fc2.weight"].T.numpy(),
                    state_dict[f"blocks.{i}.mlp.fc2.bias"].numpy(),
                ]
            )
        else:
            swiglu_fused_gate_and_value_projection_key = f"blocks.{i}.mlp.w12.weight"
            swiglu_output_projection_key = f"blocks.{i}.mlp.w3.weight"
            if swiglu_fused_gate_and_value_projection_key in state_dict:
                fused_layer_name = f"{block_prefix}_mlp_fused_gate_and_value_projection"
                output_layer_name = f"{block_prefix}_mlp_output_projection"
                keras_model.get_layer(fused_layer_name).set_weights(
                    [
                        state_dict[
                            swiglu_fused_gate_and_value_projection_key
                        ].T.numpy(),
                        state_dict[f"blocks.{i}.mlp.w12.bias"].numpy(),
                    ]
                )
                keras_model.get_layer(output_layer_name).set_weights(
                    [
                        state_dict[swiglu_output_projection_key].T.numpy(),
                        state_dict[f"blocks.{i}.mlp.w3.bias"].numpy(),
                    ]
                )
            else:
                raise KeyError(
                    f"Could not find FFN weights for block {i}. "
                    "Neither standard MLP nor SwiGLU keys were found."
                )
        layer_scale_1_key = f"blocks.{i}.ls1.gamma"
        layer_scale_2_key = f"blocks.{i}.ls2.gamma"
        ls1_layer = keras_model.get_layer(f"{block_prefix}_ls1")
        ls2_layer = keras_model.get_layer(f"{block_prefix}_ls2")
        if layer_scale_1_key in state_dict and hasattr(ls1_layer, "kernel"):
            ls1_layer.kernel.assign(state_dict[layer_scale_1_key].numpy())
        if layer_scale_2_key in state_dict and hasattr(ls2_layer, "kernel"):
            ls2_layer.kernel.assign(state_dict[layer_scale_2_key].numpy())

    logging.info("✓ Ported all transformer blocks.")

    final_normalization_layer = keras_model.get_layer("norm")
    final_normalization_layer.set_weights(
        [state_dict["norm.weight"].numpy(), state_dict["norm.bias"].numpy()]
    )
    logging.info("✓ Ported final LayerNormalization.")

    logging.info("\n--- Weight porting complete! ---")
    return keras_model


def count_blocks(keras_model):
    layer_names = [layer.name for layer in keras_model.layers]
    block_indices = [
        int(name.split("_")[1])
        for name in layer_names
        if name.startswith("block_") and name.endswith("_norm1")
    ]
    return max(block_indices) + 1


if __name__ == "__main__":
    import argparse
    import gc

    MODEL_CONSTRUCTORS = {
        "dinov2_vits14": DINOV2Small,
        "dinov2_vitb14": DINOV2Base,
        "dinov2_vitl14": DINOV2Large,
        # "dinov2_vitg14": DINOV2Giant2,
    }
    DEFAULT_VARIANTS = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"]

    parser = argparse.ArgumentParser(
        description=(
            "Port DINOv2 PyTorch weights to native Keras .keras files. "
            "The giant variant (dinov2_vitg14) is heavy (>4GB) and is OFF by "
            "default; opt in with --variants dinov2_vitg14 or --all."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(MODEL_CONSTRUCTORS.keys()),
        default=None,
        help="Subset of variants to port. Default: 3 small/base/large.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Port all variants including the giant. May exhaust RAM.",
    )
    args = parser.parse_args()

    if args.all:
        selected = list(MODEL_CONSTRUCTORS.keys())
    elif args.variants:
        selected = args.variants
    else:
        selected = DEFAULT_VARIANTS

    output_dir = "weights"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set to: '{output_dir}'")
    logging.info(f"Variants to port: {selected}")

    for model_name in selected:
        print("-" * 80)
        model_constructor = MODEL_CONSTRUCTORS[model_name]
        logging.info(f"Processing model: {model_name}")
        logging.info(f"Using Keras backend: {keras.backend.backend()}")

        logging.info("Loading pretrained DINOv2 model from Torch Hub...")
        pytorch_model = torch.hub.load(
            "facebookresearch/dinov2", model_name, verbose=False
        )
        pytorch_model.eval()
        pytorch_state_dict = {
            key: value.detach().cpu()
            for key, value in pytorch_model.state_dict().items()
        }
        del pytorch_model
        gc.collect()
        logging.info("✓ PyTorch weights extracted; model object released.")

        logging.info("Instantiating Keras 3 DINOv2 model architecture...")
        keras_dinov2_model = model_constructor(
            img_size=518,
            patch_size=14,
            init_values=1.0e-5,
            num_reg_tokens=0,
            name=f"dino_{model_name}",
        )

        port_weights_from_state_dict(keras_dinov2_model, pytorch_state_dict)
        del pytorch_state_dict
        gc.collect()

        final_keras_model_path = os.path.join(output_dir, f"{model_name}_ported.keras")
        logging.info(
            f"Saving final native Keras model to '{final_keras_model_path}'..."
        )
        keras_dinov2_model.save(final_keras_model_path)
        logging.info("✓ Save complete.")

        del keras_dinov2_model
        gc.collect()
        try:
            import jax

            jax.clear_caches()
        except Exception:
            pass
        print("-" * 80 + "\n")

    print("All requested models have been successfully ported and saved.")
