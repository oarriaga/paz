import numpy as np
import torch
from paz.models.feature.xfeat.model import XFeatModel

BLOCK_SIZES = {"block1": 4, "block2": 2, "block3": 3, "block4": 3, "block5": 4}


def port_weights(torch_path):
    state = torch.load(torch_path, map_location="cpu")
    model = XFeatModel(weights=None)
    for layer in model.layers:
        arrays = weights_for_layer(layer.name, state)
        if arrays is not None:
            layer.set_weights(arrays)
    return model


def weights_for_layer(name, state):
    if name == "skip_conv":
        return conv_bias(state, "skip1.1")
    if name.endswith("_out"):
        return conv_bias(state, output_source(name))
    if name.endswith("_conv"):
        return [conv_kernel(state[basic_source(name) + ".layer.0.weight"])]
    if name.endswith("_bn"):
        return batchnorm(state, basic_source(name))
    return None


def output_source(name):
    return {"fusion_out": "block_fusion.2",
            "heatmap_out": "heatmap_head.2",
            "keypoint_out": "keypoint_head.3"}[name]


def basic_source(name):
    stem = name.rsplit("_", 1)[0]
    prefix, index = stem.rsplit("_", 1)
    if prefix in BLOCK_SIZES:
        return f"{prefix}.{index}"
    return f"{torch_group(prefix)}.{index}"


def torch_group(prefix):
    return {"fusion": "block_fusion",
            "heatmap": "heatmap_head",
            "keypoint": "keypoint_head"}[prefix]


def conv_kernel(weight):
    return np.transpose(weight.numpy(), (2, 3, 1, 0))


def conv_bias(state, source):
    kernel = conv_kernel(state[source + ".weight"])
    return [kernel, state[source + ".bias"].numpy()]


def batchnorm(state, source):
    mean = state[source + ".layer.1.running_mean"].numpy()
    variance = state[source + ".layer.1.running_var"].numpy()
    return [mean, variance]


if __name__ == "__main__":
    import sys

    torch_path = sys.argv[1]
    output_path = sys.argv[2]
    model = port_weights(torch_path)
    model.save(output_path)
    print("saved", output_path)
