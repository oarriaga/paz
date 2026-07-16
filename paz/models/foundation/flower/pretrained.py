"""Download and build pretrained FLOWER bundles.

The converted artifact directory holds ``florence2.weights.h5`` (the
truncated Florence-2 encoder), ``flower_dit.weights.h5`` (the flow
transformer), and ``tokenizer.json``. Files larger than the GitHub
release-asset cap are sharded and reassembled with the gemma4 helpers.
"""
from collections import namedtuple
from pathlib import Path

from keras.utils import get_file

import json

from paz.models.foundation.florence2 import configuration as florence2_configuration  # fmt: skip
from paz.models.foundation.florence2 import model as florence2
from paz.models.foundation.florence2 import tokenizer as tokenizers
from paz.models.foundation.flower import configuration
from paz.models.foundation.flower import model as flow_dit
from paz.models.foundation.gemma4.pretrained import assemble_weights_file

FLOWER_WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.29/"  # fmt: skip
FLOWER_CACHE = "paz/models/flower"
FLORENCE2_VARIANT = "florence2_large_flower"
FLOWERModels = namedtuple("FLOWER", "config encoder dit tokenizer")


def FLOWERLiberoObject(weights="pretrained", models_path=None):
    return FLOWER("flower_libero_object", weights, models_path)


def FLOWER(model_name="flower_libero_object", weights="pretrained",
           models_path=None):
    config = configuration.to_config(model_name)
    encoder_config = florence2_configuration.CONFIGS[FLORENCE2_VARIANT]
    encoder = florence2.build(encoder_config)
    dit = flow_dit.build(config)
    model_dir = resolve_dir(model_name, weights, models_path)
    tokenizer_path = default_tokenizer_path(model_dir)
    tokenizer = tokenizers.load_tokenizer(tokenizer_path)
    if weights is not None or models_path is not None:
        encoder.load_weights(str(model_dir / "florence2.weights.h5"))
        dit.load_weights(str(model_dir / "flower_dit.weights.h5"))
    return FLOWERModels(config, encoder, dit, tokenizer)


def resolve_dir(model_name, weights, models_path):
    if models_path is not None:
        return Path(models_path)
    return download_weights(model_name)


def download_weights(model_name):
    subdir = "{}/{}".format(FLOWER_CACHE, model_name)
    asset = "{}.manifest.json".format(model_name)
    manifest_path = Path(get_file(
        asset, FLOWER_WEIGHTS_URL + asset, cache_subdir=subdir))
    model_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    for filename, entry in manifest.items():
        assemble_weights_file(model_dir / filename, entry, subdir)
    return model_dir


def default_tokenizer_path(model_dir):
    return model_dir / "tokenizer.json"
