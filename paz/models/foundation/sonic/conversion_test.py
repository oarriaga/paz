import numpy as np
from onnx import helper
from onnx import numpy_helper

from paz.models.foundation.sonic.conversion import apply_decoder_weights
from paz.models.foundation.sonic.conversion import apply_encoder_weights
from paz.models.foundation.sonic.model import build_sonic_decoder
from paz.models.foundation.sonic.model import build_sonic_encoder
from paz.models.foundation.sonic.model import build_toy_layout


def find_dense_layers(model, prefix):
    return [layer for layer in model.layers if layer.name.startswith(prefix)]


def build_fake_encoder_onnx(encoder, branch):
    initializers, expected = [], {}
    for layer in find_dense_layers(encoder, f"{branch}_module_"):
        layer_id = layer.name.rsplit("_", 1)[-1]
        in_dim, out_dim = layer.get_weights()[0].shape
        rng = np.random.default_rng(int(layer_id))
        weight = rng.normal(size=(out_dim, in_dim)).astype("float32")
        bias = rng.normal(size=(out_dim,)).astype("float32")
        prefix = f"module.encoders.{branch}.module.{layer_id}"
        initializers.append(numpy_helper.from_array(weight, f"{prefix}.weight"))
        initializers.append(numpy_helper.from_array(bias, f"{prefix}.bias"))
        expected[layer.name] = (weight.transpose(1, 0), bias)
    graph = helper.make_graph([], "encoder", [], [], initializer=initializers)
    return helper.make_model(graph), expected


def build_fake_decoder_onnx(decoder):
    initializers, nodes, expected = [], [], {}
    for layer in find_dense_layers(decoder, "g1_dyn_module_"):
        layer_id = layer.name.rsplit("_", 1)[-1]
        in_dim, out_dim = layer.get_weights()[0].shape
        rng = np.random.default_rng(int(layer_id))
        weight = rng.normal(size=(in_dim, out_dim)).astype("float32")
        bias = rng.normal(size=(out_dim,)).astype("float32")
        kernel_name = f"kernel_{layer_id}"
        bias_name = f"module.decoders.g1_dyn.module.{layer_id}.bias"
        initializers.append(numpy_helper.from_array(weight, kernel_name))
        initializers.append(numpy_helper.from_array(bias, bias_name))
        node_name = f"/g1_dyn/module/module.{layer_id}/MatMul"
        nodes.append(helper.make_node(
            "MatMul", ["x", kernel_name], ["y"], name=node_name))
        expected[layer.name] = (weight, bias)
    graph = helper.make_graph(
        nodes, "decoder", [], [], initializer=initializers)
    return helper.make_model(graph), expected


def test_apply_encoder_weights_matches_injected_values():
    layout = build_toy_layout()
    encoder = build_sonic_encoder(layout)
    encoder_onnx, expected = build_fake_encoder_onnx(encoder, "flat")
    apply_encoder_weights(encoder, encoder_onnx)
    for layer_name, (kernel, bias) in expected.items():
        actual_kernel, actual_bias = encoder.get_layer(layer_name).get_weights()
        assert np.array_equal(actual_kernel, kernel)
        assert np.array_equal(actual_bias, bias)


def test_apply_decoder_weights_matches_injected_values():
    layout = build_toy_layout()
    decoder = build_sonic_decoder(layout)
    decoder_onnx, expected = build_fake_decoder_onnx(decoder)
    apply_decoder_weights(decoder, decoder_onnx)
    for layer_name, (kernel, bias) in expected.items():
        actual_kernel, actual_bias = decoder.get_layer(layer_name).get_weights()
        assert np.array_equal(actual_kernel, kernel)
        assert np.array_equal(actual_bias, bias)
