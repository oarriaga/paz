"""Development-only weight conversion for DA3-SMALL (Apache-2.0).

Downloads the official checkpoint, ports it into the Keras model, saves a
``.weights.h5`` next to a metadata file, and verifies save/reload. Requires
torch, safetensors, and huggingface_hub. Not imported at runtime.

Usage: python -m paz.models.foundation.depth_anything3.convert
"""
import os
import json
import hashlib

import numpy as np

from paz.models.foundation.depth_anything3.models import build_da3_small
from paz.models.foundation.depth_anything3.models import build_da3_base
from paz.models.foundation.depth_anything3 import port_weights

UPSTREAM = "https://github.com/ByteDance-Seed/Depth-Anything-3"
UPSTREAM_COMMIT = "e74fd796e96b7e781a5506fd8503b6bd7232513c"
LICENSE = "Apache-2.0"
IMAGE_SHAPE = (518, 518, 3)
SMALL = ("depth-anything/DA3-SMALL", build_da3_small, 384, "da3_small_paz_jax")
BASE = ("depth-anything/DA3-BASE", build_da3_base, 768, "da3_base_paz_jax")


def convert(size, output_directory, image_shape=IMAGE_SHAPE, views=2):
    repository, builder, hidden_size, stem = size
    checkpoint = download_checkpoint(repository)
    state = load_state(checkpoint)
    model = build_and_port(builder, hidden_size, state, image_shape, views)
    weights_path = os.path.join(output_directory, f"{stem}.weights.h5")
    model.save_weights(weights_path)
    verify_reload(builder, hidden_size, weights_path, state, image_shape, views)
    write_metadata(output_directory, repository, stem, checkpoint, weights_path)
    return weights_path


def download_checkpoint(repository):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repository, "model.safetensors")


def load_state(checkpoint):
    from safetensors.numpy import load_file
    return load_file(checkpoint)


def build_and_port(builder, hidden_size, state, image_shape, views):
    model = builder(views, image_shape)
    positions = count_positions(image_shape)
    port_weights.port_backbone_weights(model, state, 12, positions, hidden_size)
    port_weights.port_head_weights(model, state)
    port_weights.port_camera_decoder_weights(model, state)
    return model


def verify_reload(builder, hidden_size, weights_path, state, image_shape, views):
    model = build_and_port(builder, hidden_size, state, image_shape, views)
    reloaded = builder(views, image_shape)
    reloaded.load_weights(weights_path)
    data = np.zeros((1, views, *image_shape), "float32")
    for expected, actual in zip(model(data), reloaded(data)):
        assert np.allclose(np.array(expected), np.array(actual), atol=1e-5)


def write_metadata(output_directory, repository, stem, checkpoint, weights_path):
    metadata = {"repository": repository, "upstream": UPSTREAM,
                "upstream_commit": UPSTREAM_COMMIT, "license": LICENSE,
                "source_sha256": sha256(checkpoint),
                "converted_sha256": sha256(weights_path)}
    path = os.path.join(output_directory, f"{stem}.json")
    with open(path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


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
    size = BASE if os.environ.get("DA3_MODEL") == "base" else SMALL
    print("saved:", convert(size, directory))
