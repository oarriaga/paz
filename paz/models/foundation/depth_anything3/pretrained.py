"""Download pretrained Depth Anything 3 weights from the paz data release.

Each model is a single ``.weights.h5`` asset built at the 518x518 process
resolution. See NOTICE.md for upstream attribution and licensing.
"""
from keras.utils import get_file

WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.27/"  # fmt: skip
CACHE = "paz/models/depth_anything3"
WEIGHTS = {
    "da3_small": "da3_small_paz_jax.weights.h5",
    "da3_base": "da3_base_paz_jax.weights.h5",
    "da3_mono_large": "da3_mono_large_paz_jax.weights.h5",
    "da3_metric_large": "da3_metric_large_paz_jax.weights.h5",
}


def download_weights(model_name):
    asset = WEIGHTS[model_name]
    return get_file(asset, WEIGHTS_URL + asset, cache_subdir=CACHE)
