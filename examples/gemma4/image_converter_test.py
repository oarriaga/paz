import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from .image_converter import preprocess_images
from .vision import build_vision_encoder, build_vision_encoder_args, num_patches


def test_preprocess_shapes_and_positions():
    config = build_vision_encoder_args()
    images = np.random.default_rng(0).random((2, 40, 40, 3)).astype("float32")
    out = preprocess_images(images, config)
    side = config.image_size // config.patch_size
    patch_dim = 3 * config.patch_size ** 2
    count = num_patches(config)
    assert tuple(out["pixel_values"].shape) == (2, count, patch_dim)
    assert tuple(out["pixel_position_ids"].shape) == (2, count, 2)
    positions = np.array(out["pixel_position_ids"])[0]
    assert positions[0].tolist() == [0, 0]
    assert positions[1].tolist() == [1, 0]
    assert positions[side].tolist() == [0, 1]


def test_preprocess_rectangular_pads():
    config = build_vision_encoder_args()
    images = np.random.default_rng(2).random((1, 60, 30, 3)).astype("float32")
    out = preprocess_images(images, config)
    positions = np.array(out["pixel_position_ids"])[0]
    assert tuple(out["pixel_values"].shape) == (
        1, config.max_patches, 3 * config.patch_size ** 2)
    assert (positions == -1).any()        # padded slots present
    assert (positions[0] != -1).all()     # first patch is real


def test_preprocess_feeds_vision_encoder():
    config = build_vision_encoder_args()
    encoder = build_vision_encoder(config)
    images = np.random.default_rng(1).random((1, 33, 51, 3)).astype("float32")
    out = encoder(preprocess_images(images, config))
    pooled = config.max_patches // config.pool_size ** 2
    assert tuple(np.array(out).shape) == (1, pooled, config.output_dim)
