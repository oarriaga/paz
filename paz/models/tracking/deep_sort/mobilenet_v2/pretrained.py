"""Download the MobileNetV2 DeepSORT appearance weights.

A single ``.weights.h5`` asset converted with ``port_weights.py``. See
``paz/models/tracking/deep_sort/mobilenet_v2/NOTICE.md`` for upstream
attribution and licensing.
"""
from keras.utils import get_file

RELEASES = "https://github.com/mohamedamrfaroukebrahim/paz/releases/download/"
TAG = "deep-sort-mobilenet-v1/"
WEIGHTS_URL = RELEASES + TAG
ASSET = "mobilenet_v2_paz_jax.weights.h5"
SHA256 = "9e011f4d8e75d574a90790e512420853b1b7f74c2be8ee2720dd9a579ad37213"
CACHE = "paz/models/mobilenet_v2"


def download_weights():
    kwargs = dict(cache_subdir=CACHE, file_hash=SHA256)
    return get_file(ASSET, WEIGHTS_URL + ASSET, **kwargs)
