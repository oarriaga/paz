"""Convert FLOWER checkpoint ``vlm.*`` tensors into the paz Florence-2
encoder weights.

Dev-only script: reads the FLOWER ``model.safetensors`` (numpy, no torch)
and writes ``florence2.weights.h5`` plus a keymap report. Skipped source
tensors, each with a reason verified at run time:

- ``vlm.language_shared.weight``: tied duplicate of
  ``vlm.language_encoder.embed_tokens.weight`` (verified identical).
- ``vlm.language_final_logits_bias``: bias of the deleted decoder LM
  head; the encoder-only inference path never uses it.
- ``vlm.visual_temporal_embed.pos_idx_to_embed``: deterministic cosine
  table; only row 0 (= [0, 1, 0, 1, ...]) is used for single-frame
  inference and the model computes it in code (verified equal).
"""
import re
from pathlib import Path

import numpy as np

from paz.models.foundation.florence2.model import build

STAGE_HEADS = (8, 16, 32, 64)
ENCODER_HEADS = 16
ATTENTION_ROLES = {"q": "query", "k": "key", "v": "value"}
SKIPPED = ["vlm.language_shared.weight", "vlm.language_final_logits_bias",
           "vlm.visual_temporal_embed.pos_idx_to_embed"]


def convert(checkpoint_path, output_directory):
    from safetensors.numpy import load_file
    state = load_file(checkpoint_path)
    source = {k: v for k, v in state.items() if k.startswith("vlm.")}
    verify_skipped(source)
    model = build()
    keymap = build_keymap(source)
    assign(model, source, keymap)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_directory / "florence2.weights.h5"))
    write_report(keymap, output_directory / "florence2_keymap.txt")
    return model


def verify_skipped(source):
    shared = source["vlm.language_shared.weight"]
    tokens = source["vlm.language_encoder.embed_tokens.weight"]
    if not np.array_equal(shared, tokens):
        raise ValueError("language_shared is not tied to embed_tokens")
    temporal = source["vlm.visual_temporal_embed.pos_idx_to_embed"]
    row_zero = np.tile([0.0, 1.0], temporal.shape[1] // 2)
    if not np.array_equal(temporal[0], row_zero.astype(temporal.dtype)):
        raise ValueError("temporal embedding row zero is not [0, 1, ...]")


def build_keymap(source):
    keymap = {}
    for key in sorted(source):
        if key in SKIPPED:
            continue
        keymap[key] = translate(key, source[key])
    return keymap


def translate(key, tensor):
    vision_prefix = "vlm.vision_tower."
    if key.startswith(vision_prefix):
        return translate_vision(key[len(vision_prefix):], tensor)
    if key.startswith("vlm.language_encoder.layers."):
        return translate_encoder_layer(key, tensor)
    return translate_global(key, tensor)


def translate_global(key, tensor):
    if key == "vlm.image_projection":
        return [("image_projection/kernel", tensor)]
    if key == "vlm.image_pos_embed.row_embeddings.weight":
        return [("image_row_embedding/embeddings", tensor)]
    if key == "vlm.image_pos_embed.column_embeddings.weight":
        return [("image_column_embedding/embeddings", tensor)]
    if key == "vlm.language_encoder.embed_tokens.weight":
        return [("embed_tokens/embeddings", tensor)]
    if key == "vlm.language_encoder.embed_positions.weight":
        return [("embed_positions/embeddings", tensor)]
    norms = {"vlm.image_proj_norm": "image_proj_norm",
             "vlm.language_encoder.layernorm_embedding":
                 "layernorm_embedding"}
    for prefix, layer in norms.items():
        if key.startswith(prefix + "."):
            return translate_norm(key, layer, tensor)
    raise KeyError(f"unmapped checkpoint tensor {key}")


def translate_norm(key, layer, tensor):
    role = "gamma" if key.endswith(".weight") else "beta"
    return [(f"{layer}/{role}", tensor)]


def translate_encoder_layer(key, tensor):
    pattern = r"vlm\.language_encoder\.layers\.(\d+)\.(.+)"
    layer, rest = re.match(pattern, key).groups()
    name = f"encoder_layer_{layer}"
    if rest.startswith("self_attn."):
        args = (rest[len("self_attn."):], f"{name}_self_attention",
                tensor, ENCODER_HEADS)
        return translate_attention(*args)
    if rest.startswith("self_attn_layer_norm."):
        return translate_norm(rest, f"{name}_self_attention_norm", tensor)
    if rest.startswith("final_layer_norm."):
        return translate_norm(rest, f"{name}_final_norm", tensor)
    if rest.startswith("fc1.") or rest.startswith("fc2."):
        dense, role = rest.split(".")
        value = tensor.T if role == "weight" else tensor
        role = "kernel" if role == "weight" else "bias"
        return [(f"{name}_{dense}/{role}", value)]
    raise KeyError(f"unmapped checkpoint tensor {key}")


def translate_attention(rest, name, tensor, num_heads):
    projection, role = rest.split(".")
    if projection == "out_proj":
        dim = tensor.shape[0]
        if role == "weight":
            value = tensor.T.reshape(num_heads, dim // num_heads, dim)
            return [(f"{name}_attention_output/kernel", value)]
        return [(f"{name}_attention_output/bias", tensor)]
    head = ATTENTION_ROLES[projection[0]]
    dim = tensor.shape[-1] if role == "weight" else tensor.shape[0]
    heads_shape = (num_heads, dim // num_heads)
    if role == "weight":
        value = tensor.T.reshape(dim, *heads_shape)
        return [(f"{name}_{head}/kernel", value)]
    return [(f"{name}_{head}/bias", tensor.reshape(heads_shape))]


def translate_vision(key, tensor):
    conv_match = re.match(r"convs\.(\d+)\.(proj|norm)\.(weight|bias)", key)
    if conv_match:
        return translate_stem(conv_match, tensor)
    pattern = (r"blocks\.(\d+)\.(\d+)\.(spatial|channel)_block\.(.+)")
    stage, block, kind, rest = re.match(pattern, key).groups()
    name = f"blocks_{stage}_{block}_{kind}"
    num_heads = STAGE_HEADS[int(stage)]
    if rest.startswith("conv1.fn.dw.") or rest.startswith("conv2.fn.dw."):
        conv, role = rest.split(".fn.dw.")
        value = tensor.transpose(2, 3, 0, 1) if role == "weight" else tensor
        role = "kernel" if role == "weight" else "bias"
        return [(f"{name}_{conv}/{role}", value)]
    if rest.startswith("ffn."):
        return translate_vision_ffn(rest, name, tensor)
    if kind == "spatial":
        return translate_window_attention(rest, name, tensor, num_heads)
    return translate_channel_attention(rest, name, tensor)


def translate_stem(match, tensor):
    stage, part, role = match.groups()
    if part == "norm":
        gamma = "gamma" if role == "weight" else "beta"
        return [(f"convs_{stage}_norm/{gamma}", tensor)]
    value = tensor.transpose(2, 3, 1, 0) if role == "weight" else tensor
    role = "kernel" if role == "weight" else "bias"
    return [(f"convs_{stage}_proj/{role}", value)]


def translate_vision_ffn(rest, name, tensor):
    if rest.startswith("ffn.norm."):
        return translate_norm(rest, f"{name}_ffn_norm", tensor)
    pattern = r"ffn\.fn\.net\.(fc\d)\.(weight|bias)"
    dense, role = re.match(pattern, rest).groups()
    value = tensor.T if role == "weight" else tensor
    role = "kernel" if role == "weight" else "bias"
    return [(f"{name}_ffn_{dense}/{role}", value)]


def translate_window_attention(rest, name, tensor, num_heads):
    name = f"{name}_window_attention"
    if rest.startswith("window_attn.norm."):
        return translate_norm(rest, f"{name}_norm", tensor)
    pattern = r"window_attn\.fn\.(qkv|proj)\.(weight|bias)"
    part, role = re.match(pattern, rest).groups()
    if part == "proj":
        dim = tensor.shape[0]
        if role == "weight":
            value = tensor.T.reshape(num_heads, dim // num_heads, dim)
            return [(f"{name}_attention_output/kernel", value)]
        return [(f"{name}_attention_output/bias", tensor)]
    return split_fused_qkv(tensor, role, name, num_heads)


def split_fused_qkv(tensor, role, name, num_heads):
    dim = tensor.shape[0] // 3
    heads_shape = (num_heads, dim // num_heads)
    entries = []
    for split, head in enumerate(("query", "key", "value")):
        part = tensor[split * dim:(split + 1) * dim]
        if role == "weight":
            value = part.T.reshape(dim, *heads_shape)
            entries.append((f"{name}_{head}/kernel", value))
        else:
            entry = (f"{name}_{head}/bias", part.reshape(heads_shape))
            entries.append(entry)
    return entries


def translate_channel_attention(rest, name, tensor):
    name = f"{name}_attention"
    if rest.startswith("channel_attn.norm."):
        return translate_norm(rest, f"{name}_norm", tensor)
    pattern = r"channel_attn\.fn\.(qkv|proj)\.(weight|bias)"
    part, role = re.match(pattern, rest).groups()
    value = tensor.T if role == "weight" else tensor
    role = "kernel" if role == "weight" else "bias"
    return [(f"{name}_{part}/{role}", value)]


def assign(model, source, keymap):
    variables = {v.path: v for v in model.weights}
    assigned = set()
    for key, entries in keymap.items():
        for path, value in entries:
            if path not in variables:
                raise KeyError(f"{key} maps to unknown variable {path}")
            variable = variables[path]
            if tuple(variable.shape) != value.shape:
                message = (f"{key}: shape {value.shape} does not "
                           f"match {path} {tuple(variable.shape)}")
                raise ValueError(message)
            variable.assign(value)
            assigned.add(path)
    missing = sorted(set(variables) - assigned)
    if missing:
        raise ValueError(f"unassigned model variables: {missing}")


def write_report(keymap, path):
    lines = []
    for key, entries in sorted(keymap.items()):
        paths = ", ".join(entry_path for entry_path, _ in entries)
        lines.append(f"{key} -> {paths}")
    for key in SKIPPED:
        lines.append(f"{key} -> skipped (see module docstring)")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert FLOWER vlm.*")
    add = parser.add_argument
    add("checkpoint", help="path to FLOWER model.safetensors")
    add("output", help="directory for florence2.weights.h5 and keymap")
    args = parser.parse_args()
    convert(args.checkpoint, args.output)
    print(f"saved weights and keymap to {args.output}")
