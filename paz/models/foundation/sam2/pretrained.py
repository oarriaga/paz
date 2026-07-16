"""Public SAM 2 and SAM 2.1 image-model factories and weight utilities.

The eight factories share four architectures and differ only by checkpoint.
``weights`` accepts a directory holding the four converted ``.weights.h5``
files; pass ``None`` for an uninitialized architecture. ``convert_checkpoint``
turns an official ``.pt`` into such a directory and lazily imports torch, so
the runtime never depends on it.
"""
from pathlib import Path

from paz.models.foundation.sam2 import model as sam2_model
from paz.models.foundation.sam2 import convert
from paz.models.foundation.sam2 import configuration as cfg

WEIGHT_FILES = ("image_encoder", "point_encoder", "mask_downscaling",
                "mask_decoder")


def SAM2HieraTiny(weights=None):
    return build(cfg.TINY, weights)


def SAM2HieraSmall(weights=None):
    return build(cfg.SMALL, weights)


def SAM2HieraBasePlus(weights=None):
    return build(cfg.BASE_PLUS, weights)


def SAM2HieraLarge(weights=None):
    return build(cfg.LARGE, weights)


def SAM21HieraTiny(weights=None):
    return build(cfg.TINY, weights)


def SAM21HieraSmall(weights=None):
    return build(cfg.SMALL, weights)


def SAM21HieraBasePlus(weights=None):
    return build(cfg.BASE_PLUS, weights)


def SAM21HieraLarge(weights=None):
    return build(cfg.LARGE, weights)


def build(config, weights):
    bundle = sam2_model.build(config)
    if weights is not None:
        load_weights(bundle, weights)
    return bundle


def save_weights(bundle, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for model, name in zip(bundle[:4], WEIGHT_FILES):
        model.save_weights(str(directory / f"{name}.weights.h5"))


def load_weights(bundle, directory):
    directory = Path(directory)
    for model, name in zip(bundle[:4], WEIGHT_FILES):
        model.load_weights(str(directory / f"{name}.weights.h5"))


def convert_checkpoint(checkpoint, config, directory):
    import torch
    state_dict = torch.load(checkpoint, map_location="cpu")["model"]
    arrays = {key: value.float().cpu().numpy()
              for key, value in state_dict.items()}
    bundle = sam2_model.build(config)
    convert.convert(bundle, arrays)
    save_weights(bundle, directory)
    return bundle
