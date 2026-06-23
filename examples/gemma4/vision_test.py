import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import re

import numpy as np
import keras
import pytest

from .vision import build_vision_encoder, build_vision_encoder_args, num_patches

gemma4 = pytest.importorskip(
    "keras_hub.src.models.gemma4.gemma4_vision_encoder")
Gemma4VisionEncoder = gemma4.Gemma4VisionEncoder

CONFIG = build_vision_encoder_args()


def build_reference():
    return Gemma4VisionEncoder(
        image_size=CONFIG.image_size, patch_size=CONFIG.patch_size,
        num_heads=CONFIG.num_heads, hidden_dim=CONFIG.hidden_dim,
        num_layers=CONFIG.num_layers, intermediate_dim=CONFIG.intermediate_dim,
        head_dim=CONFIG.head_dim, output_dim=CONFIG.output_dim,
        num_key_value_heads=CONFIG.num_key_value_heads,
        pool_size=CONFIG.pool_size,
        position_embedding_size=CONFIG.position_embedding_size,
        rope_max_wavelength=CONFIG.rope_wavelength,
        layer_norm_epsilon=CONFIG.layer_norm_epsilon, dropout=0.0,
        use_clipped_linears=True, standardize=False, dtype="float32")


def role(path):
    path = path.replace("/dense/", "/")
    match = re.search(r"encoder_block_(\d+)", path)
    if match:
        rest = path[match.end():].lstrip("/_").replace("/", "_")
        return "b{}:{}".format(int(match.group(1)), rest)
    if "position_embedding_table" in path:
        return "g:position_embedding_table"
    parts = path.split("/")
    return "g:{}_{}".format(parts[-2], parts[-1])


def transfer(source, target):
    weights = {role(w.path): np.asarray(keras.ops.convert_to_numpy(w))
               for w in source.weights}
    for variable in target.weights:
        variable.assign(weights[role(variable.path)].reshape(variable.shape))


def build_inputs():
    side = CONFIG.image_size // CONFIG.patch_size
    cols = np.tile(np.arange(side), side)
    rows = np.repeat(np.arange(side), side)
    positions = np.stack([cols, rows], -1).astype("int32")
    rng = np.random.default_rng(0)
    patch_dim = 3 * CONFIG.patch_size ** 2
    values = rng.standard_normal(
        (1, num_patches(CONFIG), patch_dim)).astype("float32")
    return values, positions[None]


def build_rectangular_inputs():
    rows, columns = 4, 6
    grid_x = np.tile(np.arange(columns), rows)
    grid_y = np.repeat(np.arange(rows), columns)
    positions = np.stack([grid_x, grid_y], -1).astype("int32")
    real = rows * columns
    pad = CONFIG.max_patches - real
    pad_positions = -np.ones((pad, 2), dtype="int32")
    positions = np.concatenate([positions, pad_positions], axis=0)
    patch_dim = 3 * CONFIG.patch_size ** 2
    values = np.random.default_rng(1).standard_normal(
        (1, real, patch_dim)).astype("float32")
    values = np.concatenate(
        [values, np.zeros((1, pad, patch_dim), "float32")], axis=1)
    return values, positions[None]


def compare(reference, encoder, values, positions):
    paz_out = np.array(encoder(
        {"pixel_values": values, "pixel_position_ids": positions}))
    kh_inputs = {"pixel_values": values[:, None],
                 "pixel_position_ids": positions[:, None]}
    kh_out = np.array(reference(kh_inputs)).reshape(paz_out.shape)
    return float(np.max(np.abs(paz_out - kh_out)))


def test_vision_encoder_matches_keras_hub():
    reference = build_reference()
    encoder = build_vision_encoder(CONFIG)
    transfer(reference, encoder)
    values, positions = build_inputs()
    assert compare(reference, encoder, values, positions) < 1e-4


def test_vision_encoder_matches_keras_hub_rectangular():
    reference = build_reference()
    encoder = build_vision_encoder(CONFIG)
    transfer(reference, encoder)
    values, positions = build_rectangular_inputs()
    assert compare(reference, encoder, values, positions) < 1e-4
