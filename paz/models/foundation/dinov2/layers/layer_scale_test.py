import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import keras
from keras import Input, Model
from keras.layers import EinsumDense

from paz.models.foundation.dinov2.layers.layer_scale import (
    apply_layer_scale,
    scale,
)


def build_layer_scale_model(seq_len, dim, init_values, name):
    inputs = Input(shape=(seq_len, dim), name=f"{name}_input")
    outputs = apply_layer_scale(inputs, dim, init_values, name)
    return Model(inputs=inputs, outputs=outputs, name=f"{name}_model")


def test_apply_layer_scale_none_uses_identity_layer():
    model = build_layer_scale_model(5, 8, None, "ls_none")
    layer = model.get_layer("ls_none")
    assert isinstance(layer, keras.layers.Identity)


def test_apply_layer_scale_zero_uses_identity_layer():
    model = build_layer_scale_model(5, 8, 0.0, "ls_zero")
    layer = model.get_layer("ls_zero")
    assert isinstance(layer, keras.layers.Identity)


def test_apply_layer_scale_nonzero_uses_einsum_dense():
    model = build_layer_scale_model(5, 8, 1e-5, "ls_eps")
    layer = model.get_layer("ls_eps")
    assert isinstance(layer, EinsumDense)


def test_apply_layer_scale_identity_returns_input():
    model = build_layer_scale_model(4, 6, None, "ls_idret")
    x = np.random.randn(2, 4, 6).astype(np.float32)
    y = np.array(model(x))
    np.testing.assert_array_equal(y, x)


def test_apply_layer_scale_multiplies_by_init_values():
    model = build_layer_scale_model(3, 7, 0.25, "ls_mul")
    x = np.ones((2, 3, 7), dtype=np.float32)
    y = np.array(model(x))
    expected = np.full_like(x, 0.25)
    np.testing.assert_allclose(y, expected, atol=1e-6)


def test_apply_layer_scale_kernel_shape_matches_dim():
    model = build_layer_scale_model(5, 9, 1.0, "ls_shape")
    layer = model.get_layer("ls_shape")
    assert tuple(layer.kernel.shape) == (9,)


def test_apply_layer_scale_kernel_initialized_to_init_values():
    model = build_layer_scale_model(5, 4, 0.7, "ls_init")
    layer = model.get_layer("ls_init")
    weights = np.array(layer.kernel)
    np.testing.assert_allclose(weights, np.full((4,), 0.7), atol=1e-6)


def test_apply_layer_scale_layer_name_present():
    model = build_layer_scale_model(5, 8, 1e-5, "ls_named")
    names = [layer.name for layer in model.layers]
    assert "ls_named" in names


def test_scale_assigns_kernel_and_applies_elementwise():
    inputs = Input(shape=(2, 3), name="sc_input")
    outputs = scale(inputs, 3, 1.0, "sc_layer")
    model = Model(inputs=inputs, outputs=outputs, name="sc_model")
    new_kernel = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    model.get_layer("sc_layer").kernel.assign(new_kernel)
    x = np.ones((1, 2, 3), dtype=np.float32)
    y = np.array(model(x))
    expected = np.broadcast_to(new_kernel, (1, 2, 3))
    np.testing.assert_allclose(y, expected, atol=1e-6)
