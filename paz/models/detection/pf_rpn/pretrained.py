"""Download PF-RPN weights from the paz data release.

The asset is converted from the checkpoint published by the PF-RPN authors
at https://huggingface.co/tangqh/PF-RPN (Apache-2.0).
"""
from keras.utils import get_file

WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.37/"  # fmt: skip


def download_weights(model):
    asset = f"{model.name}_paz_jax.weights.h5"
    cache = "paz/models/pf_rpn"
    return get_file(asset, WEIGHTS_URL + asset, cache_subdir=cache)
