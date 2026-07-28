"""Public SAM 2 and SAM 2.1 factories and weight utilities.

The eight image factories share four architectures and differ only by
checkpoint; the eight ``*Video`` factories add the memory modules and object
pointer projections on top of the same four. ``weights="pretrained"``
downloads and verifies the converted weights; a directory path loads local
``.weights.h5`` files; ``None`` leaves the architecture uninitialized. The
video factories reuse the hosted image weights and fetch the video-only files
alongside them. ``convert_checkpoint`` turns an official ``.pt`` into such a
directory and lazily imports torch, so the runtime never needs it.
"""
import json
from pathlib import Path

from keras.utils import get_file

from paz.models.foundation.sam2 import model as sam2_model
from paz.models.foundation.sam2 import convert
from paz.models.foundation.sam2 import configuration as cfg

ALTAMIRA = "https://github.com/oarriaga/altamira-data/releases/download/"
SAM2_IMAGE_URL = ALTAMIRA + "v0.30/"
SAM2_VIDEO_URL = ALTAMIRA + "v0.32/"
SAM2_CACHE = "paz/models/sam2"


def SAMHieraTiny2(weights="pretrained"):
    return build(cfg.TINY, "sam2_hiera_tiny", weights)


def SAMHieraSmall2(weights="pretrained"):
    return build(cfg.SMALL, "sam2_hiera_small", weights)


def SAMHieraBasePlus2(weights="pretrained"):
    return build(cfg.BASE_PLUS, "sam2_hiera_base_plus", weights)


def SAMHieraLarge2(weights="pretrained"):
    return build(cfg.LARGE, "sam2_hiera_large", weights)


def SAMHieraTiny21(weights="pretrained"):
    return build(cfg.TINY, "sam2.1_hiera_tiny", weights)


def SAMHieraSmall21(weights="pretrained"):
    return build(cfg.SMALL, "sam2.1_hiera_small", weights)


def SAMHieraBasePlus21(weights="pretrained"):
    return build(cfg.BASE_PLUS, "sam2.1_hiera_base_plus", weights)


def SAMHieraLarge21(weights="pretrained"):
    return build(cfg.LARGE, "sam2.1_hiera_large", weights)


def SAMHieraTiny2Video(weights="pretrained"):
    return build_video(cfg.TINY, "sam2_hiera_tiny", weights)


def SAMHieraSmall2Video(weights="pretrained"):
    return build_video(cfg.SMALL, "sam2_hiera_small", weights)


def SAMHieraBasePlus2Video(weights="pretrained"):
    return build_video(cfg.BASE_PLUS, "sam2_hiera_base_plus", weights)


def SAMHieraLarge2Video(weights="pretrained"):
    return build_video(cfg.LARGE, "sam2_hiera_large", weights)


def SAMHieraTiny21Video(weights="pretrained"):
    return build_video(cfg.TINY, "sam2.1_hiera_tiny", weights)


def SAMHieraSmall21Video(weights="pretrained"):
    return build_video(cfg.SMALL, "sam2.1_hiera_small", weights)


def SAMHieraBasePlus21Video(weights="pretrained"):
    return build_video(cfg.BASE_PLUS, "sam2.1_hiera_base_plus", weights)


def SAMHieraLarge21Video(weights="pretrained"):
    return build_video(cfg.LARGE, "sam2.1_hiera_large", weights)


def build(config, variant, weights):
    bundle = sam2_model.build(config)
    if weights == "pretrained":
        load_weights(bundle, download_weights(variant))
    elif weights is not None:
        load_weights(bundle, weights)
    return bundle


def build_video(config, variant, weights):
    bundle = sam2_model.build_video(config)
    if weights == "pretrained":
        load_weights(bundle, download_video_weights(variant))
    elif weights is not None:
        load_weights(bundle, weights)
    return bundle


def download_weights(variant):
    asset = f"{variant}.manifest.json"
    return download_assets(variant, asset, SAM2_IMAGE_URL)


def download_video_weights(variant):
    download_weights(variant)
    asset = f"{variant}.video.manifest.json"
    return download_assets(variant, asset, SAM2_VIDEO_URL)


def download_assets(variant, asset, url):
    subdir = f"{SAM2_CACHE}/{variant}"
    manifest = get_file(asset, url + asset, cache_subdir=subdir)
    checksums = json.loads(Path(manifest).read_text())
    for local, checksum in checksums.items():
        origin = url + f"{variant}_{local}"
        get_file(local, origin, cache_subdir=subdir, file_hash=checksum)
    return Path(manifest).parent


def save_weights(bundle, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in sam2_model.submodels(bundle):
        model.save_weights(str(directory / f"{name}.weights.h5"))


def load_weights(bundle, directory):
    directory = Path(directory)
    for name, model in sam2_model.submodels(bundle):
        model.load_weights(str(directory / f"{name}.weights.h5"))


def convert_checkpoint(checkpoint, config, directory):
    bundle = sam2_model.build(config)
    convert.convert(bundle, load_state_dict(checkpoint))
    save_weights(bundle, directory)
    return bundle


def convert_video_checkpoint(checkpoint, config, directory):
    bundle = sam2_model.build_video(config)
    convert.convert_video(bundle, load_state_dict(checkpoint))
    save_weights(bundle, directory)
    return bundle


def load_state_dict(checkpoint):
    import torch
    state_dict = torch.load(checkpoint, map_location="cpu")["model"]
    arrays = {}
    for key, value in state_dict.items():
        arrays[key] = value.float().cpu().numpy()
    return arrays
