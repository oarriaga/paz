"""Converts the released ``Kimodo-G1-RP-v1`` checkpoint to paz weights.

Development tool: maps the torch ``model.safetensors`` onto the functional
denoiser built by ``model.build`` and bundles the normalization statistics
into an ``.npz``. Every checkpoint tensor must be consumed and every shape
must agree, otherwise the conversion fails.

    python -m paz.models.foundation.kimodo.conversion \\
        --checkpoint_dir <hf snapshot of nvidia/Kimodo-G1-RP-v1> \\
        --output_dir ~/.keras/paz/models/kimodo
"""
import argparse
from pathlib import Path

import numpy as np

from paz.models.foundation.kimodo import features
from paz.models.foundation.kimodo import model as denoiser_model

PREFIX = "denoiser.backbone."
STAGES = (("root_model", "root"), ("body_model", "body"))
WEIGHTS_NAME = "kimodo_g1_rp_v1.weights.h5"
ASSETS_NAME = "kimodo_g1_rp_v1_assets.npz"


def convert(checkpoint_dir, output_dir):
    from safetensors import safe_open
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint = safe_open(str(checkpoint_dir / "model.safetensors"), "numpy")
    assets = load_assets(checkpoint_dir)
    model = denoiser_model.build(features.build_stats(assets))
    used = transfer(checkpoint, model)
    audit_keys(checkpoint, used)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_dir / WEIGHTS_NAME))
    np.savez(output_dir / ASSETS_NAME, **assets)
    (output_dir / "kimodo_g1_rp_v1_keymap.txt").write_text(build_report(used))
    return model


def load_assets(checkpoint_dir):
    assets = {}
    for block in ("global_root", "local_root", "body"):
        folder = Path(checkpoint_dir) / "stats" / "motion" / block
        assets[f"{block}_mean"] = np.load(folder / "mean.npy")
        assets[f"{block}_std"] = np.load(folder / "std.npy")
    return assets


def transfer(checkpoint, model):
    used = {}

    def tensor(key, target):
        used[PREFIX + key] = target
        return checkpoint.get_tensor(PREFIX + key)

    def set_dense(layer_name, key):
        set_linear(layer_name, f"{key}.weight", f"{key}.bias")

    def set_linear(layer_name, weight_key, bias_key):
        weights = [tensor(weight_key, layer_name).T]
        weights.append(tensor(bias_key, layer_name))
        model.get_layer(layer_name).set_weights(weights)

    def set_norm(layer_name, key):
        weights = [tensor(f"{key}.weight", layer_name)]
        weights.append(tensor(f"{key}.bias", layer_name))
        model.get_layer(layer_name).set_weights(weights)

    num_layers = count_layers(checkpoint)
    setters = (set_dense, set_linear, set_norm)
    for source, target in STAGES:
        transfer_stage(*setters, source, target, num_layers)
    return used


def transfer_stage(set_dense, set_linear, set_norm, source, target,
                   num_layers):
    set_dense(f"{target}_text_embedding", f"{source}.embed_text")
    set_dense(f"{target}_time_embedding_dense_1",
              f"{source}.embed_timestep.time_embed.0")
    set_dense(f"{target}_time_embedding_dense_2",
              f"{source}.embed_timestep.time_embed.2")
    set_dense(f"{target}_heading_embedding",
              f"{source}.linear_first_heading_angle")
    set_dense(f"{target}_input_projection", f"{source}.input_linear")
    set_dense(f"{target}_output_projection", f"{source}.output_linear")
    for layer in range(num_layers):
        key = f"{source}.seqTransEncoder.layers.{layer}"
        name = f"{target}_layer_{layer}"
        # Torch fuses query, key and value into one in_proj_weight.
        set_linear(f"{name}_self_attention_qkv",
                   f"{key}.self_attn.in_proj_weight",
                   f"{key}.self_attn.in_proj_bias")
        set_dense(f"{name}_self_attention_output", f"{key}.self_attn.out_proj")
        set_dense(f"{name}_feedforward_intermediate", f"{key}.linear1")
        set_dense(f"{name}_feedforward_output", f"{key}.linear2")
        set_norm(f"{name}_norm_1", f"{key}.norm1")
        set_norm(f"{name}_norm_2", f"{key}.norm2")


def count_layers(checkpoint):
    marker = PREFIX + "root_model.seqTransEncoder.layers."
    layers = {key[len(marker):].split(".")[0] for key in checkpoint.keys()
              if key.startswith(marker)}
    return len(layers)


def audit_keys(checkpoint, used):
    unused = sorted(set(checkpoint.keys()) - set(used))
    if unused:
        raise ValueError(f"Unmapped checkpoint tensors: {unused}")


def build_report(used):
    lines = [f"{key} -> {target}" for key, target in sorted(used.items())]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", default="~/.keras/paz/models/kimodo")
    arguments = parser.parse_args()
    convert(arguments.checkpoint_dir, Path(arguments.output_dir).expanduser())
