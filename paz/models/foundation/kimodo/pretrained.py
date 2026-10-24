"""Download and build the pretrained ``Kimodo-G1-RP-v1`` denoiser.

The converted bundle holds ``kimodo_g1_rp_v1.weights.h5`` (both stages),
``kimodo_g1_rp_v1_assets.npz`` (normalization statistics) and a small
``kimodo_g1_rp_v1_prompts.npz`` of LLM2Vec embeddings for a few prompts.
Weights are released under the NVIDIA Open Model License.
"""
from collections import namedtuple
from pathlib import Path

import numpy as np
from keras.utils import get_file

from paz.models.foundation.kimodo import features
from paz.models.foundation.kimodo import model as denoiser_model

ALTAMIRA = "https://github.com/oarriaga/altamira-data/releases/download"
KIMODO_WEIGHTS_URL = ALTAMIRA + "/v0.30/"
KIMODO_CACHE = "paz/models/kimodo"
WEIGHTS_NAME = "kimodo_g1_rp_v1.weights.h5"
ASSETS_NAME = "kimodo_g1_rp_v1_assets.npz"
PROMPTS_NAME = "kimodo_g1_rp_v1_prompts.npz"
KIMODO = namedtuple("KIMODO", "denoiser stats")


def KimodoG1RP(weights="pretrained", models_path=None):
    model_dir = resolve_dir(models_path)
    stats = features.build_stats(np.load(model_dir / ASSETS_NAME))
    denoiser = denoiser_model.build(stats)
    if weights is not None:
        denoiser.load_weights(str(model_dir / WEIGHTS_NAME))
    return KIMODO(denoiser, stats)


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
    path = get_file(name, KIMODO_WEIGHTS_URL + name, cache_subdir=KIMODO_CACHE)
    return Path(path)


def fetch_prompt_embeddings(models_path=None):
    """Loads ``{sanitized prompt: (1, 4096) embedding}`` from the bundle."""
    if models_path is None:
        path = fetch_asset(PROMPTS_NAME)
    else:
        path = Path(models_path) / PROMPTS_NAME
    with np.load(path) as prompts:
        embeddings = {text: prompts[text] for text in prompts.files}
    return embeddings
