import numpy as np
import keras
from keras import Input, Model

from paz.models.foundation.dinov2.layers.drop_path import (
    apply_drop_path,
    build_noise_shape,
)


def build_drop_path_model(seq_len, dim, rate, name):
    inputs = Input(shape=(seq_len, dim), name=f"{name}_input")
    outputs = apply_drop_path(inputs, rate, name)
    return Model(inputs=inputs, outputs=outputs, name=f"{name}_model")


def test_apply_drop_path_zero_rate_is_identity():
    model = build_drop_path_model(5, 8, 0.0, "dp_zero")
    x = np.random.randn(2, 5, 8).astype(np.float32)
    y = np.array(model(x))
    np.testing.assert_array_equal(y, x)


def test_apply_drop_path_zero_rate_uses_identity_layer():
    model = build_drop_path_model(5, 8, 0.0, "dp_id")
    layer = model.get_layer("dp_id")
    assert isinstance(layer, keras.layers.Identity)


def test_apply_drop_path_nonzero_uses_dropout_layer():
    model = build_drop_path_model(5, 8, 0.2, "dp_drop")
    layer = model.get_layer("dp_drop")
    assert isinstance(layer, keras.layers.Dropout)


def test_apply_drop_path_inference_is_identity():
    model = build_drop_path_model(4, 6, 0.5, "dp_eval")
    x = np.random.randn(3, 4, 6).astype(np.float32)
    y = np.array(model(x, training=False))
    np.testing.assert_array_equal(y, x)


def test_apply_drop_path_training_drops_whole_samples():
    keras.utils.set_random_seed(0)
    model = build_drop_path_model(4, 6, 0.5, "dp_train")
    x = np.ones((16, 4, 6), dtype=np.float32)
    y = np.array(model(x, training=True))
    per_sample = y.reshape(16, -1)
    sample_min = per_sample.min(axis=1)
    sample_max = per_sample.max(axis=1)
    np.testing.assert_array_equal(sample_min, sample_max)


def test_build_noise_shape_rank_three():
    inputs = Input(shape=(7, 9), name="ns_input")
    shape = build_noise_shape(inputs)
    assert shape == (None, 1, 1)


def test_build_noise_shape_rank_four():
    inputs = Input(shape=(7, 9, 3), name="ns4_input")
    shape = build_noise_shape(inputs)
    assert shape == (None, 1, 1, 1)


def test_drop_path_layer_name_present():
    model = build_drop_path_model(5, 8, 0.1, "dp_named")
    names = [layer.name for layer in model.layers]
    assert "dp_named" in names


def test_apply_identity_returns_input():
    inputs = Input(shape=(5, 8), name="ai_input")
    outputs = apply_drop_path(inputs, 0.0, "ai_layer")
    model = Model(inputs=inputs, outputs=outputs, name="ai_model")
    x = np.random.randn(2, 5, 8).astype(np.float32)
    y = np.array(model(x))
    np.testing.assert_array_equal(y, x)


def test_drop_path_inverted_scaling_expectation():
    keras.utils.set_random_seed(1)
    inputs = Input(shape=(4, 6), name="dpis_input")
    outputs = apply_drop_path(inputs, 0.5, "dpis_layer")
    model = Model(inputs=inputs, outputs=outputs, name="dpis_model")
    x = np.ones((2048, 4, 6), dtype=np.float32)
    y = np.array(model(x, training=True))
    mean = float(y.mean())
    assert abs(mean - 1.0) < 0.05
