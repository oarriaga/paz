import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from paz.models.transformers.embeddings import rotary


def test_apply_at_position_zero_is_identity():
    x = np.random.RandomState(0).rand(1, 3, 8).astype("float32")
    positions = np.zeros((3, 1), "float32")   # angle 0 -> cos 1, sin 0
    output = np.asarray(rotary.apply(x, 10000.0, 1.0, 8, positions))
    np.testing.assert_allclose(output, x, atol=1e-5)


def test_apply_preserves_shape():
    x = np.random.RandomState(1).rand(2, 4, 6).astype("float32")
    output = np.asarray(rotary.apply(x, 10000.0, 1.0, 6))
    assert output.shape == (2, 4, 6)


def test_apply_partial_preserves_shape():
    x = np.random.RandomState(2).rand(1, 3, 8).astype("float32")
    output = np.asarray(rotary.apply_partial(x, 10000.0, 1.0, 0.5))
    assert output.shape == (1, 3, 8)


def test_apply_2d_at_position_zero_is_identity():
    tokens = np.random.RandomState(0).randn(1, 2, 1, 8).astype("float32")
    positions = np.zeros((1, 1, 2), "int32")
    output = np.asarray(rotary.apply_2d(tokens, positions, 100.0))
    np.testing.assert_allclose(output, tokens, atol=1e-6)


def test_apply_2d_preserves_norm_of_each_axis_half():
    from paz.models.transformers.embeddings.patch import build_patch_positions
    tokens = np.random.RandomState(1).randn(2, 3, 5, 8).astype("float32")
    positions = np.broadcast_to(np.array(build_patch_positions(1, 5)), (2, 5, 2))
    output = np.asarray(rotary.apply_2d(tokens, positions, 100.0))
    for start in (0, 4):
        half = slice(start, start + 4)
        source = np.linalg.norm(tokens[..., half], axis=-1)
        rotated = np.linalg.norm(output[..., half], axis=-1)
        np.testing.assert_allclose(source, rotated, atol=1e-5)
