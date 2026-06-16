import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import keras
from keras import Input, Model

from paz.models.foundation.dinov2.layers.swiglu_ffn import (
    swiglu_ffn_fused,
    compute_fused_hidden,
)


def build_swiglu_fused_model(input_features, hidden_features, drop_rate):
    inputs = Input(shape=(None, input_features))
    args = (inputs, hidden_features, input_features, True, drop_rate, "ffn", "silu")
    outputs = swiglu_ffn_fused(*args)
    return Model(inputs, outputs, name="ffn")


def test_compute_fused_hidden_rounds_to_multiple_of_8():
    assert compute_fused_hidden(384) == 256
    assert compute_fused_hidden(1536) == 1024


def test_swiglu_ffn_fused_uses_aligned_hidden_units():
    model = build_swiglu_fused_model(48, 96, drop_rate=0.0)
    fused = model.get_layer("ffn_fused_gate_and_value_projection")
    expected_hidden = compute_fused_hidden(96)
    assert fused.units == 2 * expected_hidden


def test_swiglu_ffn_fused_output_shape_matches_input_features():
    model = build_swiglu_fused_model(48, 96, drop_rate=0.0)
    x = np.zeros((2, 5, 48), dtype="float32")
    y = model(x)
    assert tuple(y.shape) == (2, 5, 48)


def test_swiglu_ffn_fused_has_dropout_sublayer():
    model = build_swiglu_fused_model(48, 96, drop_rate=0.1)
    drop = model.get_layer("ffn_drop")
    assert isinstance(drop, keras.layers.Dropout)


def test_swiglu_ffn_fused_numerical_parity_with_manual_silu():
    model = build_swiglu_fused_model(48, 96, drop_rate=0.0)
    x = np.random.RandomState(0).randn(1, 3, 48).astype("float32")
    fused_layer = model.get_layer("ffn_fused_gate_and_value_projection")
    output_layer = model.get_layer("ffn_output_projection")
    fused = fused_layer(x)
    value, gate = np.split(np.array(fused), 2, axis=-1)
    activated = value * (1.0 / (1.0 + np.exp(-value)))
    expected = np.array(output_layer(activated * gate))
    actual = np.array(model(x))
    assert np.allclose(actual, expected, atol=1e-5)
