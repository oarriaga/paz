import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import keras
import pytest

from paz.models.foundation.gemma4.conversion import role as text_role
from paz.models.foundation.gemma4.model import build_text_backbone_args
from paz.models.foundation.gemma4.multimodal import build_multimodal_backbone
from paz.models.foundation.gemma4.vision import (
    build_vision_encoder_args, num_patches)

backbone = pytest.importorskip(
    "keras_hub.src.models.gemma4.gemma4_backbone")
vision_module = pytest.importorskip(
    "keras_hub.src.models.gemma4.gemma4_vision_encoder")
Gemma4Backbone = backbone.Gemma4Backbone
Gemma4VisionEncoder = vision_module.Gemma4VisionEncoder

from paz.models.foundation.gemma4.vision_test import role as vision_role

TEXT = build_text_backbone_args(num_layers=2, sliding_window_pattern=2)
VISION = build_vision_encoder_args(output_dim=TEXT.hidden_dim)


def build_reference():
    vision = Gemma4VisionEncoder(
        image_size=VISION.image_size, patch_size=VISION.patch_size,
        num_heads=VISION.num_heads, hidden_dim=VISION.hidden_dim,
        num_layers=VISION.num_layers, intermediate_dim=VISION.intermediate_dim,
        head_dim=VISION.head_dim, output_dim=VISION.output_dim,
        num_key_value_heads=VISION.num_key_value_heads,
        pool_size=VISION.pool_size,
        position_embedding_size=VISION.position_embedding_size,
        rope_max_wavelength=VISION.rope_wavelength,
        layer_norm_epsilon=VISION.layer_norm_epsilon, dtype="float32")
    return Gemma4Backbone(
        vocabulary_size=TEXT.vocabulary_size, image_size=VISION.image_size,
        num_layers=TEXT.num_layers, num_query_heads=TEXT.num_query_heads,
        num_key_value_heads=TEXT.num_key_value_heads,
        hidden_dim=TEXT.hidden_dim, intermediate_dim=TEXT.intermediate_dim,
        head_dim=TEXT.head_dim, sliding_window_size=TEXT.sliding_window_size,
        sliding_window_pattern=TEXT.sliding_window_pattern,
        vision_encoder=vision, layer_norm_epsilon=TEXT.layer_norm_epsilon,
        dtype="float32")


def role(path):
    if "decoder_block_" in path or "token_embedding" in path \
            or "final_normalization" in path or "per_layer" in path:
        return "t:" + text_role(path)
    return "v:" + vision_role(path)


def transfer(source, target):
    weights = {role(w.path): np.asarray(keras.ops.convert_to_numpy(w))
               for w in source.weights}
    for variable in target.weights:
        variable.assign(weights[role(variable.path)].reshape(variable.shape))


def build_inputs():
    batch, seq = 2, 12
    side = VISION.image_size // VISION.patch_size
    pooled = (side // VISION.pool_size) ** 2
    rng = np.random.default_rng(0)
    tokens = rng.integers(1, TEXT.vocabulary_size, (batch, seq)).astype("int32")
    padding = np.ones((batch, seq), dtype="int32")
    positions = np.broadcast_to(np.arange(seq, dtype="int32"), (batch, seq))
    cols = np.tile(np.arange(side), side)
    rows = np.repeat(np.arange(side), side)
    grid = np.stack([cols, rows], -1).astype("int32")
    grid = np.broadcast_to(grid, (batch,) + grid.shape)
    patch_dim = 3 * VISION.patch_size ** 2
    pixels = rng.standard_normal(
        (batch, num_patches(VISION), patch_dim)).astype("float32")
    vision_indices = np.tile(
        np.arange(2, 2 + pooled, dtype="int32"), (batch, 1))
    return tokens, padding, positions.copy(), grid, pixels, vision_indices


def test_multimodal_matches_keras_hub():
    reference = build_reference()
    model = build_multimodal_backbone(TEXT, VISION)
    transfer(reference, model)
    tokens, padding, positions, grid, pixels, indices = build_inputs()
    paz_out = np.array(model({
        "token_ids": tokens, "padding_mask": padding, "pixel_values": pixels,
        "pixel_position_ids": grid, "vision_indices": indices}))
    kh_out = np.array(reference({
        "token_ids": tokens, "padding_mask": padding,
        "position_ids": positions, "vision_mask": np.zeros_like(tokens),
        "vision_indices": indices, "pixel_values": pixels[:, None],
        "pixel_position_ids": grid[:, None]}))
    assert float(np.max(np.abs(paz_out - kh_out))) < 1e-4
