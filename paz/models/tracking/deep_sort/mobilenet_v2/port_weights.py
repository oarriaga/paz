"""Ports the PyTorch MobileNetV2 state dict onto the paz model.

Needs ``torch`` to read the checkpoint, so this module is imported by the
converter and the tests, never by the model itself.

    python -m paz.models.tracking.deep_sort.mobilenet_v2.port_weights \
        mobilenetv2_bottleneck_wts.pt mobilenet_v2_paz_jax.weights.h5
"""
import sys
from collections import namedtuple

import numpy as np
import torch
from keras.layers import BatchNormalization, DepthwiseConv2D

from paz.models.tracking.deep_sort.mobilenet_v2.model import MobileNetV2

Report = namedtuple("Report", "used skipped unused")
Assignment = namedtuple("Assignment", "layer_name keys arrays")

NORM_FIELDS = ("weight", "bias", "running_mean", "running_var")


def convert(torch_path, output_path):
    model = MobileNetV2(weights=None)
    report = port_weights(model, torch_path)
    if len(report.unused) > 0:
        raise ValueError(f"torch keys left unported: {report.unused}")
    model.save_weights(output_path)
    return report


def port_weights(model, torch_path):
    state = load_state(torch_path)
    assignments = build_assignments(model, state)
    for assignment in assignments:
        model.get_layer(assignment.layer_name).set_weights(assignment.arrays)
    return build_report(state, assignments)


def load_state(torch_path):
    state = torch.load(torch_path, map_location="cpu")
    return {key: value.numpy() for key, value in state.items()}


def build_assignments(model, state):
    assignments = []
    for layer in model.layers:
        if layer.weights:
            assignments.append(build_assignment(layer, state))
    return assignments


def build_report(state, assignments):
    used = [key for assignment in assignments for key in assignment.keys]
    skipped = [key for key in state if key.endswith(".num_batches_tracked")]
    accounted = set(used + skipped)
    unused = [key for key in state if key not in accounted]
    return Report(used, skipped, unused)


def build_assignment(layer, state):
    source = find_source(layer.name)
    if isinstance(layer, BatchNormalization):
        keys = [f"{source}.{field}" for field in NORM_FIELDS]
        arrays = convert_batch_normalization(state, source)
    else:
        keys = [f"{source}.weight"]
        arrays = [convert_convolution(layer, state[keys[0]])]
    return Assignment(layer.name, keys, arrays)


def convert_convolution(layer, weight):
    if isinstance(layer, DepthwiseConv2D):
        kernel = convert_depthwise_kernel(weight)
    else:
        kernel = convert_kernel(weight)
    return kernel


def find_source(layer_name):
    parts = layer_name.split("_")
    if parts[0] == "block":
        source = find_block_source(int(parts[1]), parts[2], parts[3])
    else:
        module_arg = {"stem": 0, "head": 18}[parts[0]]
        source = f"features.{module_arg}.{find_offset(parts[1])}"
    return source


def find_block_source(block_arg, unit, kind):
    # block 0 has no expansion, so its torch sequence starts at depthwise
    units = ["expand", "depthwise", "project"][int(block_arg == 0):]
    position = 3 * units.index(unit) + find_offset(kind)
    return f"features.{block_arg + 1}.conv.{position}"


def find_offset(kind):
    return ["conv", "bn"].index(kind)


def convert_kernel(weight):
    return np.transpose(weight, (2, 3, 1, 0))


def convert_depthwise_kernel(weight):
    return np.transpose(weight, (2, 3, 0, 1))


def convert_batch_normalization(state, source):
    return [state[f"{source}.{field}"] for field in NORM_FIELDS]


if __name__ == "__main__":
    print(convert(sys.argv[1], sys.argv[2]))
