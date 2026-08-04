import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from paz.models.transformers import conditioning


def test_modulate_broadcasts_over_sequence():
    x = np.ones((2, 3, 4), "float32")
    shift = np.full((2, 4), 2.0, "float32")
    scale = np.full((2, 4), 0.5, "float32")
    output = np.asarray(conditioning.modulate(x, shift, scale))
    assert np.allclose(output, 1.0 * 1.5 + 2.0)


def test_zero_modulation_is_identity():
    x = np.random.default_rng(0).normal(size=(2, 3, 4)).astype("float32")
    zeros = np.zeros((2, 4), "float32")
    output = np.asarray(conditioning.modulate(x, zeros, zeros))
    assert np.allclose(output, x)


def test_gate_scales_per_batch_channel():
    x = np.ones((2, 3, 4), "float32")
    values = np.zeros((2, 4), "float32")
    values[1] = 2.0
    output = np.asarray(conditioning.gate(x, values))
    assert np.allclose(output[0], 0.0)
    assert np.allclose(output[1], 2.0)
