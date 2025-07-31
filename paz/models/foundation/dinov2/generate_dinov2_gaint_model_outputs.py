import os
import torch
import numpy as np
import keras
from typing import Dict

os.environ["KERAS_BACKEND"] = "jax"

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
import sys

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from paz.models.foundation.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
    BlockChunk,
)
from paz.models.foundation.dinov2.layers.block import NestedTensorBlock
from paz.models.foundation.dinov2.layers.attention import Attention
from paz.models.foundation.dinov2.layers.swiglu_ffn import SwiGLUFFNFused
from paz.models.foundation.dinov2.layers.layer_scale import LayerScale
from paz.models.foundation.dinov2.layers.drop_path import DropPath

MODEL_CONFIG = {
    "name": "dinov2_vitg14",
    "keras_path": "weights/dinov2_vitg14_ported.keras",
    "pytorch_name": "dinov2_vitg14",
}
DEFAULT_INPUT_SIZE = 518
PYTORCH_HUB_REPO = "facebookresearch/dinov2"
OUTPUT_DIR = "test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)



def get_test_input(size: int = DEFAULT_INPUT_SIZE, file_path: str = None):
    """Generates or loads a consistent test input."""
    if file_path and os.path.exists(file_path):
        print(f"Loading existing test input from {file_path}")
        with np.load(file_path) as data:
            return torch.from_numpy(data["torch_input"]), data["keras_input"]

    print("Generating new random test input...")
    torch.manual_seed(0)  # for reproducibility
    torch_input = torch.randn(1, 3, size, size)
    keras_input = torch_input.permute(0, 2, 3, 1).numpy()

    if file_path:
        np.savez(file_path, torch_input=torch_input.numpy(), keras_input=keras_input)
        print(f"Saved test input to {file_path}")

    return torch_input, keras_input


def extract_pytorch_intermediates(model, input_tensor) -> Dict[str, np.ndarray]:
    """Extracts intermediate outputs from the PyTorch model."""
    print("Extracting PyTorch intermediate outputs...")
    outputs = {}
    hooks = []

    def get_hook(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            outputs[name] = output.detach().cpu().numpy()

        return hook

    hooks.append(model.patch_embed.register_forward_hook(get_hook("patch_embed")))
    hooks.append(model.norm.register_forward_hook(get_hook("norm")))
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(get_hook(f"blocks.{i}")))
        hooks.append(block.norm1.register_forward_hook(get_hook(f"blocks.{i}.norm1")))
        hooks.append(block.attn.register_forward_hook(get_hook(f"blocks.{i}.attn")))
        hooks.append(block.ls1.register_forward_hook(get_hook(f"blocks.{i}.ls1")))
        hooks.append(block.norm2.register_forward_hook(get_hook(f"blocks.{i}.norm2")))
        hooks.append(block.mlp.register_forward_hook(get_hook(f"blocks.{i}.mlp")))
        hooks.append(block.ls2.register_forward_hook(get_hook(f"blocks.{i}.ls2")))
        if hasattr(block.mlp, "w12"):
            hooks.append(
                block.mlp.w12.register_forward_hook(
                    get_hook(f"blocks.{i}.mlp.fused_gate_and_value_projection")
                )
            )
        if hasattr(block.mlp, "w3"):
            hooks.append(
                block.mlp.w3.register_forward_hook(
                    get_hook(f"blocks.{i}.mlp.output_projection")
                )
            )
        if hasattr(block.mlp, "act"):
            hooks.append(
                block.mlp.act.register_forward_hook(
                    get_hook(f"blocks.{i}.mlp.activation")
                )
            )

    with torch.no_grad():
        final_output = model(input_tensor).detach().cpu().numpy()

    outputs["final_output"] = final_output
    for hook in hooks:
        hook.remove()
    return outputs


def perform_keras_step_by_step_forward(
    keras_model, keras_input: np.ndarray
) -> Dict[str, np.ndarray]:
    """Performs a detailed step-by-step forward pass through the Keras model."""
    print("Performing Keras step-by-step forward pass...")
    keras_outputs = {}


    raw_patch_embed = keras_model.patch_embedding(keras_input)
    keras_outputs["patch_embed"] = raw_patch_embed

    B, H, W, C = keras.ops.shape(keras_input)
    x = raw_patch_embed

    classification_tokens = keras.ops.broadcast_to(
        keras_model.classification_token, (B, 1, keras_model.embedding_dimension)
    )
    x = keras.ops.concatenate([classification_tokens, x], axis=1)

    x = keras.ops.add(x, keras_model.interpolate_positional_encoding(x, H, W))

    if (
        hasattr(keras_model, "register_tokens")
        and keras_model.register_tokens is not None
    ):
        register_tokens = keras.ops.broadcast_to(
            keras_model.register_tokens,
            (B, keras_model.number_of_register_tokens, keras_model.embedding_dimension),
        )
        x = keras.ops.concatenate([x[:, :1], register_tokens, x[:, 1:]], axis=1)


    for i, chunk in enumerate(keras_model.blocks):
        for j, block in enumerate(chunk.blocks):
            block_idx = i * len(chunk.blocks) + j

            normalized_x_1 = block.normalization1(x)
            attention_output = block.attention(normalized_x_1, training=False)
            scaled_attention = block.layer_scale_1(attention_output, training=False)
            x = keras.ops.add(x, scaled_attention)

            keras_outputs[f"blocks.{block_idx}.norm1"] = normalized_x_1
            keras_outputs[f"blocks.{block_idx}.attn"] = attention_output
            keras_outputs[f"blocks.{block_idx}.ls1"] = scaled_attention

            normalized_x_2 = block.normalization2(x)

            mlp = block.mlp
            gate_and_value = mlp.fused_gate_and_value_projection(normalized_x_2)
            value, gate = keras.ops.split(gate_and_value, 2, axis=-1)
            activated_value = mlp.activation_layer(value)
            hidden = activated_value * gate
            mlp_output = mlp.output_projection(hidden)

            scaled_mlp = block.layer_scale_2(mlp_output, training=False)
            x = keras.ops.add(x, scaled_mlp)

            keras_outputs[f"blocks.{block_idx}.norm2"] = normalized_x_2
            keras_outputs[f"blocks.{block_idx}.mlp.fused_gate_and_value_projection"] = (
                gate_and_value
            )
            keras_outputs[f"blocks.{block_idx}.mlp.activation"] = activated_value
            keras_outputs[f"blocks.{block_idx}.mlp.output_projection"] = mlp_output
            keras_outputs[f"blocks.{block_idx}.mlp"] = mlp_output
            keras_outputs[f"blocks.{block_idx}.ls2"] = scaled_mlp
            keras_outputs[f"blocks.{block_idx}"] = x

    final_norm_out = keras_model.get_layer("norm")(x)
    keras_outputs["norm"] = final_norm_out
    keras_outputs["final_output"] = final_norm_out[:, 0]

    return keras_outputs


if __name__ == "__main__":
    input_file_path = os.path.join(OUTPUT_DIR, "test_input.npz")
    torch_input, keras_input = get_test_input(file_path=input_file_path)

    print("\n--- Processing PyTorch Model ---")
    pytorch_model = torch.hub.load(PYTORCH_HUB_REPO, MODEL_CONFIG["pytorch_name"])
    pytorch_model.eval()
    pytorch_outputs = extract_pytorch_intermediates(pytorch_model, torch_input)
    pytorch_output_path = os.path.join(OUTPUT_DIR, "pytorch_outputs.npz")
    np.savez(pytorch_output_path, **pytorch_outputs)
    print(f"✅ PyTorch outputs saved to {pytorch_output_path}")
    del pytorch_model, pytorch_outputs

    print("\n--- Processing Keras Model ---")
    custom_objects = {
        "DinoVisionTransformer": DinoVisionTransformer,
        "BlockChunk": BlockChunk,
        "NestedTensorBlock": NestedTensorBlock,
        "Attention": Attention,
        "LayerScale": LayerScale,
        "DropPath": DropPath,
        "SwiGLUFFNFused": SwiGLUFFNFused,
    }
    keras_model = keras.models.load_model(
        MODEL_CONFIG["keras_path"], custom_objects=custom_objects
    )
    keras_outputs = perform_keras_step_by_step_forward(keras_model, keras_input)

    keras_final_only = np.array(keras_model(keras_input, training=False))
    keras_outputs["final_output_from_full_model"] = keras_final_only

    keras_output_path = os.path.join(OUTPUT_DIR, "keras_outputs.npz")
    keras_outputs_np = {k: np.array(v) for k, v in keras_outputs.items()}
    np.savez(keras_output_path, **keras_outputs_np)
    print(f"✅ Keras outputs saved to {keras_output_path}")
    del keras_model, keras_outputs, keras_outputs_np

    print("\n🎉 All outputs generated successfully!")
