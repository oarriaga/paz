"""Download and build pretrained Gemma4 bundles.

One canonical text artifact (`backbone.weights.h5`) holds all transformer
weights; the vision encoder stays a separate artifact. GitHub release assets are
capped at 2 GB, so larger files are uploaded as byte-identical parts and
reassembled on download (see `shard_weights`).
"""
import hashlib
import json
import shutil
from collections import namedtuple
from pathlib import Path

import jax.numpy as jp
from keras.utils import get_file

from paz.models.foundation.gemma4.configuration import load_config
from paz.models.foundation.gemma4.causal_lm import Gemma4CausalLM
from paz.models.foundation.gemma4.vision import VisionEncoderArgs
from paz.models.foundation.gemma4.vision import build_vision_encoder

GEMMA4_WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.26/"  # fmt: skip
GEMMA4_CACHE = "paz/models/gemma4"
PART_BYTES = 1_900_000_000
GEMMA4_WEIGHT_FILES = (
    "config.json", "tokenizer.json", "vision_config.json",
    "vision_encoder.weights.h5", "backbone.weights.h5",
)
Gemma4Models = namedtuple("Gemma4", "config model vision_encoder")


def Gemma4(model_name="gemma4_2b", weights="pretrained", models_path=None):
    model_dir = resolve_dir(model_name, models_path)
    config = load_config(model_dir / "config.json")
    model = build_causal_lm(config)
    vision_encoder = build_vision_encoder_from_dir(model_dir)
    if weights is not None:
        model.backbone.load_weights(str(model_dir / "backbone.weights.h5"))
        load_vision_weights(model_dir, vision_encoder)
    return Gemma4Models(config, model, vision_encoder)


def build_causal_lm(config):
    model = Gemma4CausalLM(config)
    materialize_weights(model, config)
    return model


def materialize_weights(model, config):
    # Subclassed models create variables lazily; run one tiny forward so
    # load_weights has variables to fill. A seq_len of 2 keeps attention's
    # key axis above size 1, avoiding a degenerate all-ones softmax.
    token_ids = jp.zeros((1, 2), dtype="int32")
    padding_mask = jp.ones((1, 2), dtype="int32")
    model({"token_ids": token_ids, "padding_mask": padding_mask})


def resolve_dir(model_name, models_path):
    if models_path is not None:
        return Path(models_path)
    return download_weights(model_name)


def download_weights(model_name):
    subdir = "{}/{}".format(GEMMA4_CACHE, model_name)
    asset = "{}.manifest.json".format(model_name)
    manifest_path = Path(get_file(
        asset, GEMMA4_WEIGHTS_URL + asset, cache_subdir=subdir))
    model_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    for filename, entry in manifest.items():
        assemble_weights_file(model_dir / filename, entry, subdir)
    return model_dir


def assemble_weights_file(path, entry, subdir):
    checksum = entry.get("sha256")
    if is_complete(path, checksum):
        return path
    parts = []
    for asset in entry["parts"]:
        parts.append(get_file(
            asset, GEMMA4_WEIGHTS_URL + asset, cache_subdir=subdir))
    concatenate_parts(parts, path)
    if checksum is not None and compute_sha256(path) != checksum:
        raise ValueError("Checksum mismatch after assembling {}".format(path))
    return path


def is_complete(path, checksum):
    if not path.exists():
        return False
    return checksum is None or compute_sha256(path) == checksum


def concatenate_parts(parts, output):
    with open(str(output), "wb") as merged:
        for part in parts:
            with open(str(part), "rb") as chunk:
                shutil.copyfileobj(chunk, merged)
    return output


def compute_sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_weights(source_dir, model_name, output_dir, part_bytes=PART_BYTES):
    """Split a local weights dir into <2 GB parts plus an upload manifest."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename in GEMMA4_WEIGHT_FILES:
        source = source_dir / filename
        if not source.exists():
            continue
        prefix = "{}_{}".format(model_name, filename)
        parts = split_file(source, output_dir, prefix, part_bytes)
        manifest[filename] = {"parts": parts, "sha256": compute_sha256(source)}
    manifest_path = output_dir / "{}.manifest.json".format(model_name)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def split_file(source, output_dir, prefix, part_bytes=PART_BYTES):
    output_dir = Path(output_dir)
    parts, index = [], 0
    with open(str(source), "rb") as file:
        while True:
            block = file.read(part_bytes)
            if not block:
                break
            asset = "{}.part{}".format(prefix, index)
            (output_dir / asset).write_bytes(block)
            parts.append(asset)
            index = index + 1
    return parts


def build_vision_encoder_from_dir(model_dir):
    path = Path(model_dir) / "vision_config.json"
    if not path.exists():
        return None
    with open(str(path), encoding="utf-8") as file:
        config = VisionEncoderArgs(**json.load(file))
    return build_vision_encoder(config)


def load_vision_encoder(model_dir, weights="pretrained"):
    """Build and load just the vision encoder, on the current default device.

    Use inside a `jax.default_device(...)` block to place it where you want,
    then swap it into a bundle: `Gemma4(...)._replace(vision_encoder=vision)`.
    """
    model_dir = Path(model_dir)
    vision_encoder = build_vision_encoder_from_dir(model_dir)
    if weights is not None and vision_encoder is not None:
        vision_encoder.load_weights(
            str(model_dir / "vision_encoder.weights.h5"))
    return vision_encoder


def load_vision_weights(model_dir, vision_encoder):
    if vision_encoder is not None:
        vision_encoder.load_weights(
            str(Path(model_dir) / "vision_encoder.weights.h5"))
