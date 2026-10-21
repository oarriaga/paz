import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import functools

import keras
import numpy as np

from paz.models.tracking.deep_sort.mobilenet_v2.model import MobileNetV2

BLOCK_SPECS = (
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
)
RESIDUAL_BLOCKS = (2, 4, 5, 7, 8, 9, 11, 12, 14, 15)


def build_block_configs():
    configs, in_channels = [], 32
    for expansion, out_channels, num_repeats, stride in BLOCK_SPECS:
        for repeat_arg in range(num_repeats):
            block_stride = stride if repeat_arg == 0 else 1
            config = (in_channels, expansion, out_channels, block_stride)
            configs.append(config)
            in_channels = out_channels
    return configs


def list_units(expansion):
    units = ["expand", "depthwise", "project"]
    return units[1:] if expansion == 1 else units


def build_expected_names():
    names = ["stem_conv", "stem_bn"]
    for block_arg, config in enumerate(build_block_configs()):
        for unit in list_units(config[1]):
            unit_name = f"block_{block_arg}_{unit}"
            names.extend([f"{unit_name}_conv", f"{unit_name}_bn"])
    return names + ["head_conv", "head_bn"]


def build_expected_activations():
    names = ["stem_relu", "head_relu"]
    for block_arg, config in enumerate(build_block_configs()):
        for unit in list_units(config[1])[:-1]:
            names.append(f"block_{block_arg}_{unit}_relu")
    return names


@functools.cache
def build_model():
    return MobileNetV2(weights=None)


@functools.cache
def build_random_model():
    return randomize_weights(MobileNetV2(weights=None), 0)


def randomize_weights(model, seed):
    # default initializers drive every feature to ~1e-11, which would make
    # any feature comparison pass for a wrong model
    random = np.random.RandomState(seed)
    for layer in model.layers:
        if layer.weights:
            shapes = [array.shape for array in layer.get_weights()]
            class_name = type(layer).__name__
            layer.set_weights(sample_weights(random, class_name, shapes))
    return model


def sample_weights(random, class_name, shapes):
    if class_name == "BatchNormalization":
        num_channels = shapes[0][0]
        beta = random.uniform(0.0, 0.2, num_channels)
        arrays = [np.ones(num_channels), beta, np.zeros(num_channels)]
        arrays = arrays + [np.ones(num_channels)]
    else:
        arrays = [sample_kernel(random, class_name, shapes[0])]
    return [array.astype(np.float32) for array in arrays]


def sample_kernel(random, class_name, shape):
    fan_in = shape[0] * shape[1]
    if class_name != "DepthwiseConv2D":
        fan_in = fan_in * shape[2]
    return random.randn(*shape) * np.sqrt(2.0 / fan_in)


def assert_informative(features):
    assert np.max(np.abs(features)) > 1e-2
    assert np.mean(np.std(features, axis=0)) > 1e-2


def find_block_output(block_arg):
    is_residual = block_arg in RESIDUAL_BLOCKS
    suffix = "add" if is_residual else "project_bn"
    return f"block_{block_arg}_{suffix}"


def count_layers(model, class_name):
    names = [type(layer).__name__ for layer in model.layers]
    return names.count(class_name)


def select_layers(model, class_names):
    selected = []
    for layer in model.layers:
        if type(layer).__name__ in class_names:
            selected.append(layer)
    return selected


def build_ramp(height, width, num_channels):
    rows = np.arange(height)[:, None, None]
    columns = 3.0 * np.arange(width)[None, :, None]
    channels = np.arange(num_channels)[None, None, :]
    return ((rows + columns + channels) / height).astype(np.float32)


def convolve_reference(x, kernel, stride):
    padded = np.pad(x.astype(np.float64), ((1, 1), (1, 1), (0, 0)))
    height, width = x.shape[0] // stride, x.shape[1] // stride
    output = 0.0
    for row in range(3):
        for column in range(3):
            rows = slice(row, row + stride * height, stride)
            columns = slice(column, column + stride * width, stride)
            output = output + padded[rows, columns] @ kernel[row, column]
    return output


def convolve_depthwise_reference(x, kernel, stride):
    padded = np.pad(x.astype(np.float64), ((1, 1), (1, 1), (0, 0)))
    height, width = x.shape[0] // stride, x.shape[1] // stride
    output = 0.0
    for row in range(3):
        for column in range(3):
            rows = slice(row, row + stride * height, stride)
            columns = slice(column, column + stride * width, stride)
            patch = padded[rows, columns]
            output = output + patch * kernel[row, column, :, 0]
    return output


def apply_padded_layer(model, name, x, kernel):
    convolution = model.get_layer(f"{name}_conv")
    convolution.set_weights([kernel])
    padded = model.get_layer(f"{name}_pad")(x[None])
    return np.asarray(convolution(padded))[0]


def test_model_maps_images_to_pooled_features():
    model = build_model()
    images = np.random.RandomState(0).randn(2, 224, 224, 3)
    features = np.asarray(model(images.astype(np.float32)))
    assert tuple(model.input_shape) == (None, 224, 224, 3)
    assert tuple(model.output_shape) == (None, 1280)
    assert features.shape == (2, 1280)
    assert features.dtype == np.float32


def test_layer_inventory():
    model = build_model()
    assert count_layers(model, "Conv2D") == 35
    assert count_layers(model, "DepthwiseConv2D") == 17
    assert count_layers(model, "BatchNormalization") == 52
    assert count_layers(model, "Add") == 10


def test_weighted_layers_have_stable_names():
    weighted_names = []
    for layer in build_model().layers:
        if layer.weights:
            weighted_names.append(layer.name)
    assert len(weighted_names) == len(set(weighted_names)) == 104
    assert set(weighted_names) == set(build_expected_names())


def test_residual_blocks_add_their_input():
    add_layers = select_layers(build_model(), ("Add",))
    expected = {f"block_{block_arg}_add" for block_arg in RESIDUAL_BLOCKS}
    assert {layer.name for layer in add_layers} == expected


def test_batch_normalization_uses_torch_epsilon():
    norms = select_layers(build_model(), ("BatchNormalization",))
    assert len(norms) == 52
    for norm in norms:
        assert norm.epsilon == 1e-5


def test_convolutions_have_no_bias():
    class_names = ("Conv2D", "DepthwiseConv2D")
    for convolution in select_layers(build_model(), class_names):
        assert convolution.use_bias is False


def test_three_by_three_convolutions_pad_symmetrically():
    model = build_model()
    class_names = ("Conv2D", "DepthwiseConv2D")
    for convolution in select_layers(model, class_names):
        if tuple(convolution.kernel_size) == (3, 3):
            name = convolution.name.removesuffix("_conv")
            padding = model.get_layer(f"{name}_pad").padding
            assert convolution.padding == "valid"
            assert tuple(map(tuple, padding)) == ((1, 1), (1, 1))


def test_stem_matches_symmetric_stride_two_reference():
    model = MobileNetV2(weights=None)
    kernel = np.random.RandomState(1).randn(3, 3, 3, 32).astype(np.float32)
    ramp = build_ramp(224, 224, 3)
    output = apply_padded_layer(model, "stem", ramp, kernel)
    expected = convolve_reference(ramp, kernel, 2)
    assert output.shape == (112, 112, 32)
    assert np.allclose(output, expected, rtol=0, atol=1e-4)


def test_depthwise_matches_symmetric_stride_two_reference():
    model = MobileNetV2(weights=None)
    kernel = np.random.RandomState(2).randn(3, 3, 96, 1).astype(np.float32)
    ramp = build_ramp(112, 112, 96)
    output = apply_padded_layer(model, "block_1_depthwise", ramp, kernel)
    expected = convolve_depthwise_reference(ramp, kernel, 2)
    assert output.shape == (56, 56, 96)
    assert np.allclose(output, expected, rtol=0, atol=1e-4)


def test_activations_clip_at_six():
    model = build_model()
    values = np.array([[-1.0, 3.0, 6.5, 7.0]], np.float32)
    for name in build_expected_activations():
        output = np.asarray(model.get_layer(name)(values))
        assert np.array_equal(output, [[0.0, 3.0, 6.0, 6.0]])
    assert len(build_expected_activations()) == 35


def test_features_are_spatial_mean_of_head():
    model = build_random_model()
    head = keras.Model(model.input, model.get_layer("head_relu").output)
    images = np.random.RandomState(3).randn(4, 224, 224, 3)
    images = images.astype(np.float32)
    activations = np.asarray(head(images))
    features = np.asarray(model(images))
    assert_informative(features)
    assert activations.shape == (4, 7, 7, 1280)
    assert np.allclose(features, activations.mean(axis=(1, 2)), atol=1e-5)


def test_residual_adds_block_input_to_projection():
    model = build_random_model()
    names = []
    for block_arg in RESIDUAL_BLOCKS:
        names.append(find_block_output(block_arg - 1))
        names.append(f"block_{block_arg}_project_bn")
        names.append(f"block_{block_arg}_add")
    outputs = [model.get_layer(name).output for name in names]
    images = np.random.RandomState(4).randn(2, 224, 224, 3)
    values = keras.Model(model.input, outputs)(images.astype(np.float32))
    for arg in range(0, len(names), 3):
        block_input, projection, added = values[arg:arg + 3]
        assert_informative(np.asarray(block_input).reshape(2, -1))
        assert np.allclose(added, block_input + projection, atol=1e-5)
