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
