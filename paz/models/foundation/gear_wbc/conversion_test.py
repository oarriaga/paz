import numpy as np
import pytest

pytest.importorskip("onnx")

from onnx import helper
from onnx import numpy_helper

from paz.models.foundation.gear_wbc.conversion import apply_actor_weights
from paz.models.foundation.gear_wbc.model import build_actor


def find_dense_layers(actor):
    return [layer for layer in actor.layers if layer.get_weights()]


def build_fake_actor_onnx(actor):
    initializers, expected = [], {}
    for layer in find_dense_layers(actor):
        stack, layer_index = layer.name.rsplit("_", 1)
        in_dim, out_dim = layer.get_weights()[0].shape
        rng = np.random.default_rng(len(initializers))
        weight = rng.normal(size=(out_dim, in_dim)).astype("float32")
        bias = rng.normal(size=(out_dim,)).astype("float32")
        prefix = f"{stack}.{layer_index}"
        weight_name, bias_name = f"{prefix}.weight", f"{prefix}.bias"
        initializers.append(numpy_helper.from_array(weight, weight_name))
        initializers.append(numpy_helper.from_array(bias, bias_name))
        expected[layer.name] = (weight.transpose(1, 0), bias)
    graph = helper.make_graph([], "actor", [], [], initializer=initializers)
    return helper.make_model(graph), expected


def test_apply_actor_weights_matches_injected_values():
    actor = build_actor()
    actor_onnx, expected = build_fake_actor_onnx(actor)
    apply_actor_weights(actor, actor_onnx)
    for layer_name, (kernel, bias) in expected.items():
        actual_kernel, actual_bias = actor.get_layer(layer_name).get_weights()
        assert np.array_equal(actual_kernel, kernel)
        assert np.array_equal(actual_bias, bias)


def test_apply_actor_weights_covers_every_dense_layer():
    actor = build_actor()
    _, expected = build_fake_actor_onnx(actor)
    assert len(expected) == 7
