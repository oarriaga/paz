"""Migrate the old split Gemma4 weights into one backbone.weights.h5.

The published v0.24 assets (decoder_step.weights.h5 + embedding_step.weights.h5)
were serialized from the old functional step models. Their h5 groups use
class-auto names, so recovering the semantic role of each tensor needs the old
model structure. This runs in two phases:

  --phase dump   (run on the PRE-REFACTOR commit): loads the split weights into
                 the old Gemma4DecoderStep + Gemma4PerLayerEmbeddingStep and
                 saves each variable to an .npy keyed by its role.
  --phase build  (run on this refactor): builds the new Gemma4Backbone and fills
                 it from the role dump, then writes backbone.weights.h5.

Keras-hub users do not need this; convert with conversion.convert instead. This
path exists to re-export the already-downloaded weights offline.
"""
import argparse
import re
from pathlib import Path

import numpy as np
from keras import ops


def role_key(path):
    match = re.search(r"decoder_block_(\d+)", path)
    if match:
        rest = path[match.end():].lstrip("/_").replace("/", "_")
        rest = rest.replace("attention_attention_output", "attention_output")
        if "layer_scalar" in rest:
            rest = "layer_scalar"
        return "b{}:{}".format(int(match.group(1)), rest)
    return "g:" + "_".join(path.split("/")[-2:])


def dump_phase(model_dir, dump_dir):
    from paz.models.foundation.gemma4.configuration import load_config
    from paz.models.foundation.gemma4.model import (
        Gemma4DecoderStep, Gemma4PerLayerEmbeddingStep)
    config = load_config(model_dir / "config.json")
    decoder = Gemma4DecoderStep(config)
    decoder.load_weights(str(model_dir / "decoder_step.weights.h5"))
    embedding = Gemma4PerLayerEmbeddingStep(config)
    embedding.load_weights(str(model_dir / "embedding_step.weights.h5"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    for model in (decoder, embedding):
        for weight in model.weights:
            array = np.asarray(ops.convert_to_numpy(weight))
            np.save(dump_dir / (safe(role_key(weight.path)) + ".npy"), array)


def build_phase(model_dir, dump_dir):
    import ml_dtypes
    from paz.models.foundation.gemma4.configuration import load_config
    from paz.models.foundation.gemma4.model import Gemma4Backbone
    import jax.numpy as jp
    config = load_config(model_dir / "config.json")
    backbone = Gemma4Backbone(config)
    backbone({"token_ids": jp.zeros((1, 1), "int32"),
              "padding_mask": jp.ones((1, 1), "int32")})
    for weight in backbone.weights:
        array = np.load(dump_dir / (safe(role_key(weight.path)) + ".npy"))
        if array.dtype == np.dtype("V2"):
            array = array.view(ml_dtypes.bfloat16)
        weight.assign(array.reshape(weight.shape))
    backbone.save_weights(str(model_dir / "backbone.weights.h5"))


def safe(key):
    return key.replace(":", "__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Migrate Gemma4 split weights")
    add = parser.add_argument
    add("--phase", required=True, choices=("dump", "build"))
    add("--model_dir", required=True)
    add("--dump_dir", required=True)
    args = parser.parse_args()
    model_dir, dump_dir = Path(args.model_dir), Path(args.dump_dir)
    if args.phase == "dump":
        dump_phase(model_dir, dump_dir)
    else:
        build_phase(model_dir, dump_dir)
