"""Import released SONIC ONNX weights into the ported Keras models."""

from argparse import ArgumentParser
from pathlib import Path
import re

import onnx
from onnx import numpy_helper

from paz.models.foundation.sonic.layout import compute_decoder_input_dim
from paz.models.foundation.sonic.layout import compute_encoder_input_dim
from paz.models.foundation.sonic.layout import load_release_observation_layout
from paz.models.foundation.sonic.model import build_actor
from paz.models.foundation.sonic.model import build_decoder
from paz.models.foundation.sonic.model import build_encoder

_ENCODER_BRANCH_PATTERN = re.compile(
    r"^module\.encoders\.(?P<branch>[^.]+)\.module\.(?P<layer>\d+)"
    r"\.(?P<kind>weight|bias)$")
_DECODER_NODE_PATTERN = re.compile(
    r"^/g1_dyn/module/module\.(?P<layer>\d+)/MatMul$")


def port_weights(layout, encoder_onnx_path, decoder_onnx_path):
    encoder = build_encoder(layout)
    decoder = build_decoder(layout)
    check_input_dim(encoder, compute_encoder_input_dim(layout), "encoder")
    check_input_dim(decoder, compute_decoder_input_dim(layout), "decoder")
    apply_encoder_weights(encoder, load_onnx_model(encoder_onnx_path))
    apply_decoder_weights(decoder, load_onnx_model(decoder_onnx_path))
    actor = build_actor(layout, encoder, decoder)
    return encoder, decoder, actor


def check_input_dim(model, expected_dim, name):
    input_dim = model.input_shape[-1]
    if input_dim != expected_dim:
        raise ValueError(f"{name} dim mismatch: {input_dim} != {expected_dim}")


def load_onnx_model(model_path):
    return onnx.load(str(model_path))


def load_onnx_initializers(model):
    initializers = {}
    for tensor in model.graph.initializer:
        initializers[tensor.name] = numpy_helper.to_array(tensor)
    return initializers


def apply_encoder_weights(encoder, encoder_onnx):
    initializers = load_onnx_initializers(encoder_onnx)
    for name, value in initializers.items():
        match = _ENCODER_BRANCH_PATTERN.match(name)
        if match is None:
            continue
        branch, layer_id = match.group("branch"), match.group("layer")
        layer_name = f"{branch}_module_{layer_id}"
        layer = encoder.get_layer(layer_name)
        kind = match.group("kind")
        set_dense_weight(layer, kind, value)


def set_dense_weight(layer, kind, value):
    kernel, bias = layer.get_weights()
    if kind == "weight":
        kernel = value.transpose(1, 0)
    else:
        bias = value
    layer.set_weights([kernel, bias])


def apply_decoder_weights(decoder, decoder_onnx):
    initializers = load_onnx_initializers(decoder_onnx)
    kernel_names = find_decoder_kernel_names(decoder_onnx)
    for layer_id, kernel_name in kernel_names.items():
        layer = decoder.get_layer(f"g1_dyn_module_{layer_id}")
        bias_name = f"module.decoders.g1_dyn.module.{layer_id}.bias"
        weights = [initializers[kernel_name], initializers[bias_name]]
        layer.set_weights(weights)


def find_decoder_kernel_names(decoder_onnx):
    kernel_names = {}
    for node in decoder_onnx.graph.node:
        match = _DECODER_NODE_PATTERN.match(node.name)
        if match is not None:
            kernel_names[int(match.group("layer"))] = node.input[1]
    return kernel_names


def save_ported_weights(encoder, decoder, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_weights(output_dir / "sonic_encoder.weights.h5")
    decoder.save_weights(output_dir / "sonic_decoder.weights.h5")
    return output_dir


def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument("--obs_config", required=True)
    parser.add_argument("--encoder_onnx", required=True)
    parser.add_argument("--decoder_onnx", required=True)
    parser.add_argument("--output_dir", default="~/.keras/paz/models/sonic")
    return parser


def main():
    args = build_argument_parser().parse_args()
    layout = load_release_observation_layout(args.obs_config)
    encoder_onnx, decoder_onnx = args.encoder_onnx, args.decoder_onnx
    encoder, decoder, _ = port_weights(layout, encoder_onnx, decoder_onnx)
    output_dir = Path(args.output_dir).expanduser()
    output_dir = save_ported_weights(encoder, decoder, output_dir)
    print(f"Saved ported SONIC weights to {output_dir}")


if __name__ == "__main__":
    main()
