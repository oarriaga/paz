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
from keras import ops

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.gemma4.configuration import save_config
from examples.gemma4.inference import (Gemma4DecoderStep,
                                       Gemma4PerLayerEmbeddingStep)
from examples.gemma4.model import TextBackboneArgs

ROLE_SYNONYMS = {
    "per_layer_token_embedding": "per_layer_embeddings",
    "per_layer_input_gate": "per_layer_gate",
    "per_layer_up_proj": "per_layer_projection",
    "post_per_layer_input_norm": "post_per_layer_norm",
}


def convert(preset, output_dir):
    from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
    backbone = Gemma4Backbone.from_preset(preset)
    return save_paz_models(backbone, output_dir)


def save_paz_models(backbone, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_paz_config(backbone)
    save_config(config, output_dir / "config.json")
    decoder_step = Gemma4DecoderStep(config)
    transfer(backbone, decoder_step)
    decoder_step.save_weights(str(output_dir / "decoder_step.weights.h5"))
    embedding_step = Gemma4PerLayerEmbeddingStep(config)
    transfer(backbone, embedding_step)
    embedding_step.save_weights(str(output_dir / "embedding_step.weights.h5"))
    return config


def transfer(source, target):
    weights = {role(w.path): to_numpy(w) for w in source.weights}
    for variable in target.weights:
        key = role(variable.path)
        if key not in weights:
            raise KeyError("no source weight for {}".format(variable.path))
        variable.assign(weights[key].reshape(variable.shape))


def role(path):
    match = re.search(r"decoder_block_(\d+)", path)
    if match:
        rest = path[match.end():].lstrip("/_").replace("/", "_")
        if "layer_scalar" in rest:
            rest = "layer_scalar"
        return "b{}:{}".format(int(match.group(1)), canonical(rest))
    return "g:" + canonical(path.replace("/", "_"))


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
        "num_kv_shared_layers"
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
