"""Development-only weight conversion for Depth Anything 3.

Downloads the official checkpoint, ports it into the Keras model, saves a
``.weights.h5`` next to a metadata file, and verifies save/reload. Requires
torch, safetensors, and huggingface_hub. Not imported at runtime.

Usage: python -m paz.models.foundation.depth_anything3.convert
"""
import os
import json
import hashlib

import numpy as np

from paz.models.foundation.depth_anything3 import models
from paz.models.foundation.depth_anything3 import port_weights

UPSTREAM = "https://github.com/ByteDance-Seed/Depth-Anything-3"
UPSTREAM_COMMIT = "e74fd796e96b7e781a5506fd8503b6bd7232513c"
LICENSE = "Apache-2.0"
IMAGE_SHAPE = (518, 518, 3)
SMALL = ("depth-anything/DA3-SMALL", models.DepthAnything3Small, 384,
         "da3_small_paz_jax")
BASE = ("depth-anything/DA3-BASE", models.DepthAnything3Base, 768,
        "da3_base_paz_jax")
MONO = ("depth-anything/DA3MONO-LARGE", models.DepthAnything3MonoLarge,
        "da3_mono_large_paz_jax")
METRIC = ("depth-anything/DA3METRIC-LARGE", models.DepthAnything3MetricLarge,
          "da3_metric_large_paz_jax")


def convert(size, output_directory, image_shape=IMAGE_SHAPE, views=2):
    repository, builder, hidden_size, stem = size
    state = load_checkpoint(repository)
    model = build_and_port(builder, hidden_size, state, image_shape, views)
    path = os.path.join(output_directory, f"{stem}.weights.h5")
    model.save_weights(path)
    verify_reload(model, builder(views, image_shape), path, image_shape, views)
    write_metadata(output_directory, repository, stem, state, path)
    return path


def convert_mono(size, output_directory, image_shape=IMAGE_SHAPE):
    repository, builder, stem = size
    state = load_checkpoint(repository)
    model = build_and_port_mono(builder, state, image_shape)
    path = os.path.join(output_directory, f"{stem}.weights.h5")
    model.save_weights(path)
    verify_reload(model, builder(image_shape), path, image_shape, None)
    write_metadata(output_directory, repository, stem, state, path)
    return path


def build_and_port(builder, hidden_size, state, image_shape, views):
    model = builder(views, image_shape)
    positions = count_positions(image_shape)
    port_weights.port_backbone_weights(model, state, 12, positions, hidden_size)
    port_weights.port_head_weights(model, state)
    port_weights.port_camera_decoder_weights(model, state)
    return model


def build_and_port_mono(builder, state, image_shape):
    model = builder(image_shape)
    positions = count_positions(image_shape)
    args = model, state, 24, positions, 1024
    port_weights.port_backbone_weights(*args, use_camera=False,
                                       use_qk_norm=False)
    port_weights.port_dpt_head_weights(model, state)
    return model


def verify_reload(model, reloaded, path, image_shape, views):
    reloaded.load_weights(path)
    shape = (1, *image_shape) if views is None else (1, views, *image_shape)
    data = np.zeros(shape, "float32")
    for expected, actual in zip(model(data), reloaded(data)):
        assert np.allclose(np.array(expected), np.array(actual), atol=1e-5)


def write_metadata(output_directory, repository, stem, state, path):
    checkpoint = find_checkpoint(repository)
    metadata = dict(repository=repository, upstream=UPSTREAM,
                    upstream_commit=UPSTREAM_COMMIT, license=LICENSE,
                    source_sha256=sha256(checkpoint),
                    converted_sha256=sha256(path))
    with open(os.path.join(output_directory, f"{stem}.json"), "w") as opened:
        json.dump(metadata, opened, indent=2)


def load_checkpoint(repository):
    from safetensors.numpy import load_file
    return load_file(find_checkpoint(repository))


def find_checkpoint(repository):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repository, "model.safetensors")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as opened:
        for chunk in iter(lambda: opened.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_positions(image_shape):
    return (image_shape[0] // 14) * (image_shape[1] // 14) + 1


if __name__ == "__main__":
    directory = os.environ.get("DA3_OUTPUT", ".")
    selection = os.environ.get("DA3_MODEL", "small")
    table = dict(small=SMALL, base=BASE, mono=MONO, metric=METRIC)
    size = table[selection]
    convert_size = convert_mono if selection in ("mono", "metric") else convert
    print("saved:", convert_size(size, directory))
