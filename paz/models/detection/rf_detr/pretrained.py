"""Download COCO RF-DETR weights from the paz data release.

Each variant is a single ``.weights.h5`` asset converted from the published
Roboflow checkpoint at that variant's training resolution. See NOTICE.md for
upstream attribution and licensing.
"""
from keras.utils import get_file

WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.35/"  # fmt: skip
CACHE = "paz/models/rf_detr"
WEIGHTS = {
    "rf_detr_nano": "rf_detr_nano_coco_paz_jax.weights.h5",
    "rf_detr_small": "rf_detr_small_coco_paz_jax.weights.h5",
    "rf_detr_medium": "rf_detr_medium_coco_paz_jax.weights.h5",
    "rf_detr_base": "rf_detr_base_coco_paz_jax.weights.h5",
    "rf_detr_large": "rf_detr_large_coco_paz_jax.weights.h5",
}


def download_weights(model):
    asset = WEIGHTS[model.name]
    return get_file(asset, WEIGHTS_URL + asset, cache_subdir=CACHE)
