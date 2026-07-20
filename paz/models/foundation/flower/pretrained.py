"""Download and build pretrained FLOWER bundles.

The converted artifact directory holds ``florence2.weights.h5`` (the
truncated Florence-2 encoder), ``flower_dit.weights.h5`` (the flow
transformer), and ``tokenizer.json``. Files larger than the GitHub
release-asset cap are sharded and reassembled with the gemma4 helpers.
"""
import json
from collections import namedtuple
from pathlib import Path

from keras.utils import get_file

from paz.models.foundation.florence2 import model as florence2
from paz.models.foundation.florence2 import tokenizer as tokenizers
from paz.models.foundation.flower import model as flow_dit
from paz.models.foundation.gemma4.pretrained import assemble_weights_file

ALTAMIRA = "https://github.com/oarriaga/altamira-data/releases/download"
FLOWER_WEIGHTS_URL = ALTAMIRA + "/v0.29/"
FLOWER_CACHE = "paz/models/flower"
FLOWERModels = namedtuple("FLOWER", "encoder dit tokenizer")


def FLOWERLiberoObject(weights="pretrained", models_path=None):
    return FLOWER("flower_libero_object", weights, models_path)


def FLOWER(model_name="flower_libero_object", weights="pretrained",
           models_path=None):
    encoder = florence2.build()
    dit = flow_dit.build()
    model_dir = resolve_dir(model_name, models_path)
    tokenizer = tokenizers.load_tokenizer(model_dir / "tokenizer.json")
    if weights is not None:
        encoder.load_weights(str(model_dir / "florence2.weights.h5"))
        dit.load_weights(str(model_dir / "flower_dit.weights.h5"))
    return FLOWERModels(encoder, dit, tokenizer)


def resolve_dir(model_name, models_path):
    if models_path is not None:
        return Path(models_path)
    return download_weights(model_name)


def download_weights(model_name):
    subdir = "{}/{}".format(FLOWER_CACHE, model_name)
    asset = "{}.manifest.json".format(model_name)
    url = FLOWER_WEIGHTS_URL + asset
    manifest_path = Path(get_file(asset, url, cache_subdir=subdir))
    model_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    for filename, entry in manifest.items():
        args = (model_dir / filename, entry, subdir, FLOWER_WEIGHTS_URL)
        assemble_weights_file(*args)
    return model_dir
