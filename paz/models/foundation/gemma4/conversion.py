"""Convert a keras_hub Gemma4 text backbone into paz weight files.

Run this with keras_hub's Gemma4 available (keras>=3.13 and the keras-hub
source on PYTHONPATH); the paz runtime itself does not need keras_hub Gemma4.
It writes the split inference artifacts that demo_e2b.py loads: config.json,
decoder_step.weights.h5 and embedding_step.weights.h5.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jp
from keras import ops

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paz.models.foundation.gemma4.configuration import save_config
from paz.models.foundation.gemma4.model import Gemma4Backbone
from paz.models.foundation.gemma4.configuration import TextBackboneArgs

ROLE_SYNONYMS = {
    "per_layer_token_embedding": "per_layer_embeddings",
    "per_layer_input_gate": "per_layer_gate",
    "per_layer_up_proj": "per_layer_projection",
    "post_per_layer_input_norm": "post_per_layer_norm",
    "attention_attention_output": "attention_output",
}


def convert(preset, output_dir):
    from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
    # Gemma4 checkpoints are natively bfloat16; loading as bf16 is lossless and
    # halves peak host memory versus the float32 default, so the larger E4B
    # backbone plus the paz target both fit on a 31 GB host.
    backbone = Gemma4Backbone.from_preset(preset, dtype="bfloat16")
    return save_paz_models(backbone, output_dir)


def save_paz_models(source, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_paz_config(source)
    save_config(config, output_dir / "config.json")
    backbone = build_target_backbone(config)
    transfer(source, backbone)
    backbone.save_weights(str(output_dir / "backbone.weights.h5"))
    return config


def build_target_backbone(config):
    backbone = Gemma4Backbone(config)
    backbone({"token_ids": jp.zeros((1, 1), "int32"),
              "padding_mask": jp.ones((1, 1), "int32")})
    return backbone


def transfer(source, target):
    # Map roles to source variables by reference (no bulk numpy copy), then
    # convert one weight at a time so peak memory stays near a single tensor.
    sources = {role(w.path): w for w in source.weights}
    for variable in target.weights:
        key = role(variable.path)
        if key not in sources:
            raise KeyError("no source weight for {}".format(variable.path))
        variable.assign(to_numpy(sources[key]).reshape(variable.shape))


def role(path):
    match = re.search(r"decoder_block_(\d+)", path)
    if match:
        rest = path[match.end():].lstrip("/_").replace("/", "_")
        if "layer_scalar" in rest:
            rest = "layer_scalar"
        return "b{}:{}".format(int(match.group(1)), canonical(rest))
    # Globals: key on the owning layer + variable only, so a subclassed
    # model-name prefix (e.g. "gemma4_backbone/") does not shift the role.
    leaf = "_".join(path.split("/")[-2:])
    return "g:" + canonical(leaf)


def canonical(name):
    for source_name, target_name in ROLE_SYNONYMS.items():
        name = name.replace(source_name, target_name)
    return name


def to_numpy(variable):
    return np.asarray(ops.convert_to_numpy(variable))


def build_paz_config(backbone):
    config = backbone.get_config()
    return TextBackboneArgs(**read_text_config(config))


def read_text_config(config):
    keys = (
        "vocabulary_size image_size num_layers num_query_heads "
        "num_key_value_heads hidden_dim intermediate_dim head_dim "
        "attention_logit_soft_cap final_logit_soft_cap "
        "use_sliding_window_attention sliding_window_size "
        "sliding_window_pattern global_head_dim "
        "local_rope_scaling_factor global_rope_scaling_factor "
        "global_rope_partial_rotary_factor use_bidirectional_attention "
        "layer_norm_epsilon dropout hidden_size_per_layer_input "
        "num_kv_shared_layers use_double_wide_mlp"
    ).split()
    values = {key: config[key] for key in keys}
    values["dtype"] = extract_dtype(config)
    values["local_rope_wavelength"] = config.get("local_rope_wavelength") \
        or 10_000.0
    values["global_rope_wavelength"] = config.get("global_rope_wavelength") \
        or 1_000_000.0
    values["global_layer_indices"] = None
    return values


def extract_dtype(config):
    dtype = config.get("dtype", "float32")
    if isinstance(dtype, dict):
        return dtype.get("config", {}).get("name", "float32")
    return dtype


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert keras_hub Gemma4 to paz")
    add = parser.add_argument
    add("--preset", required=True)
    add("--output_dir", required=True)
    args = parser.parse_args()
    convert(args.preset, args.output_dir)
