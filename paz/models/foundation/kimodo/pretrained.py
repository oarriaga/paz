"""Download and build the pretrained ``Kimodo-G1-RP-v1`` denoiser.

The converted bundle holds ``kimodo_g1_rp_v1.weights.h5`` (both stages)
and ``kimodo_g1_rp_v1_assets.npz`` (normalization statistics). Weights
are released under the NVIDIA Open Model License.
"""
from collections import namedtuple
from pathlib import Path

import numpy as np
from keras.utils import get_file

from paz.models.foundation.kimodo import features
from paz.models.foundation.kimodo import model as denoiser_model

ALTAMIRA = "https://github.com/oarriaga/altamira-data/releases/download"
WEIGHTS_URL = ALTAMIRA + "/v0.30/"
CACHE_SUBDIR = "paz/models/kimodo"
WEIGHTS_NAME = "kimodo_g1_rp_v1.weights.h5"
ASSETS_NAME = "kimodo_g1_rp_v1_assets.npz"
KIMODO = namedtuple("KIMODO", "denoiser stats")


def KimodoG1RP(models_path=None):
    """Pretrained denoiser and statistics; downloads unless a path is given."""
    model_dir = resolve_dir(models_path)
    kimodo = build_from_assets(model_dir)
    kimodo.denoiser.load_weights(str(model_dir / WEIGHTS_NAME))
    return kimodo


def build_from_assets(model_dir):
    stats = features.build_stats(np.load(Path(model_dir) / ASSETS_NAME))
    return KIMODO(denoiser_model.build(stats), stats)


def resolve_dir(models_path):
    if models_path is None:
        model_dir = download_assets()
    else:
        model_dir = Path(models_path)
    return model_dir


def download_assets():
    paths = [fetch_asset(name) for name in (WEIGHTS_NAME, ASSETS_NAME)]
    return paths[0].parent


def fetch_asset(name):
    return Path(get_file(name, WEIGHTS_URL + name, cache_subdir=CACHE_SUBDIR))
