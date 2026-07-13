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

from paz.models.foundation.dinov2_legacy.layers.attention import (
    split_query_key_value,
    compute_scores,
    apply_attention,
    merge_heads,
    flatten_heads,
)

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "dinov2_vitg14"
MODEL_CONFIG = {
    "name": MODEL_NAME,
    "keras_path": f"weights/{MODEL_NAME}_ported.keras",
    "pytorch_name": MODEL_NAME,
}
VARIANT_NUM_HEADS = {
    "dinov2_vits14": 6,
    "dinov2_vitb14": 12,
    "dinov2_vitl14": 16,
    "dinov2_vitg14": 24,
}
DEFAULT_INPUT_SIZE = 518
PYTORCH_HUB_REPO = "facebookresearch/dinov2"
OUTPUT_DIR = "test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def count_blocks(keras_model):
    indices = []
    for layer in keras_model.layers:
        name = layer.name
        if name.startswith("block_") and name.endswith("_norm1"):
            indices.append(int(name.split("_")[1]))
    return max(indices) + 1


def has_layer(keras_model, name):
    try:
        keras_model.get_layer(name)
        return True
    except Exception:
        return False



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

    embed_dim = keras_model.get_layer("cls_token").embeddings.shape[-1]
    depth = count_blocks(keras_model)
    num_heads = VARIANT_NUM_HEADS[MODEL_NAME]
    head_dim = embed_dim // num_heads
    scale = head_dim**-0.5
    patch_size = 14
    is_swiglu = has_layer(
        keras_model, "block_0_mlp_fused_gate_and_value_projection"
    )

    proj_layer = keras_model.get_layer("patch_embed_proj")
    projected = proj_layer(keras_input)
    B = keras.ops.shape(keras_input)[0]
    H = keras.ops.shape(keras_input)[1]
    W = keras.ops.shape(keras_input)[2]
    num_patches = (H // patch_size) * (W // patch_size)
    raw_patch_embed = keras.ops.reshape(projected, (B, num_patches, embed_dim))
    keras_outputs["patch_embed"] = raw_patch_embed

    x = raw_patch_embed

    cls_table = keras_model.get_layer("cls_token").embeddings
    classification_tokens = keras.ops.broadcast_to(
        keras.ops.expand_dims(cls_table, axis=0), (B, 1, embed_dim)
    )
    x = keras.ops.concatenate([classification_tokens, x], axis=1)

    pos_table = keras_model.get_layer("pos_embed").embeddings
    x = keras.ops.add(x, keras.ops.expand_dims(pos_table, axis=0))

    try:
        register_layer = keras_model.get_layer("register_tokens")
    except Exception:
        register_layer = None
    if register_layer is not None:
        register_table = register_layer.embeddings
        num_register = register_table.shape[0]
        register_tokens = keras.ops.broadcast_to(
            keras.ops.expand_dims(register_table, axis=0),
            (B, num_register, embed_dim),
        )
        x = keras.ops.concatenate([x[:, :1], register_tokens, x[:, 1:]], axis=1)

    for block_idx in range(depth):
        prefix = f"block_{block_idx}"

        norm1_layer = keras_model.get_layer(f"{prefix}_norm1")
        qkv_layer = keras_model.get_layer(f"{prefix}_qkv")
        proj_out_layer = keras_model.get_layer(f"{prefix}_proj")
        ls1_layer = keras_model.get_layer(f"{prefix}_ls1")
        norm2_layer = keras_model.get_layer(f"{prefix}_norm2")
        ls2_layer = keras_model.get_layer(f"{prefix}_ls2")

        normalized_x_1 = norm1_layer(x)
        qkv_out = qkv_layer(normalized_x_1)
        q, k, v = split_query_key_value(qkv_out, num_heads, head_dim)
        scores = compute_scores(q, k, scale)
        attended = apply_attention(scores, v, 0.0, name=prefix)
        merged = merge_heads(attended)
        flat = flatten_heads(merged)
        attention_output = proj_out_layer(flat)
        scaled_attention = ls1_layer(attention_output)
        x = keras.ops.add(x, scaled_attention)

        keras_outputs[f"blocks.{block_idx}.norm1"] = normalized_x_1
        keras_outputs[f"blocks.{block_idx}.attn"] = attention_output
        keras_outputs[f"blocks.{block_idx}.ls1"] = scaled_attention

        normalized_x_2 = norm2_layer(x)

        if is_swiglu:
            fused_layer = keras_model.get_layer(
                f"{prefix}_mlp_fused_gate_and_value_projection"
            )
            output_proj_layer = keras_model.get_layer(
                f"{prefix}_mlp_output_projection"
            )
            gate_and_value = fused_layer(normalized_x_2)
            value, gate = keras.ops.split(gate_and_value, 2, axis=-1)
            activated_value = keras.activations.silu(value)
            hidden = activated_value * gate
            mlp_output = output_proj_layer(hidden)
            keras_outputs[
                f"blocks.{block_idx}.mlp.fused_gate_and_value_projection"
            ] = gate_and_value
            keras_outputs[f"blocks.{block_idx}.mlp.activation"] = activated_value
            keras_outputs[f"blocks.{block_idx}.mlp.output_projection"] = mlp_output
        else:
            fc1_layer = keras_model.get_layer(f"{prefix}_mlp_fc1")
            act_layer = keras_model.get_layer(f"{prefix}_mlp_act")
            fc2_layer = keras_model.get_layer(f"{prefix}_mlp_fc2")
            hidden_pre = fc1_layer(normalized_x_2)
            hidden = act_layer(hidden_pre)
            mlp_output = fc2_layer(hidden)
            keras_outputs[f"blocks.{block_idx}.mlp.activation"] = hidden

        scaled_mlp = ls2_layer(mlp_output)
        x = keras.ops.add(x, scaled_mlp)

        keras_outputs[f"blocks.{block_idx}.norm2"] = normalized_x_2
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
    keras_model = keras.models.load_model(MODEL_CONFIG["keras_path"])
    keras_outputs = perform_keras_step_by_step_forward(keras_model, keras_input)

    keras_final_only = np.array(keras_model(keras_input, training=False))
    keras_outputs["final_output_from_full_model"] = keras_final_only

    keras_output_path = os.path.join(OUTPUT_DIR, "keras_outputs.npz")
    keras_outputs_np = {k: np.array(v) for k, v in keras_outputs.items()}
    np.savez(keras_output_path, **keras_outputs_np)
    print(f"✅ Keras outputs saved to {keras_output_path}")
    del keras_model, keras_outputs, keras_outputs_np

    print("\n🎉 All outputs generated successfully!")
