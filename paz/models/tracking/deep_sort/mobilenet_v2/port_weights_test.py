import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import functools
import hashlib
import importlib.util

import jax
import numpy as np
import pytest

from paz.models.tracking.deep_sort.mobilenet_v2.model import MobileNetV2
from paz.models.tracking.deep_sort.mobilenet_v2 import model_test
from paz.models.tracking.deep_sort.mobilenet_v2 import pretrained

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from paz.models.tracking.deep_sort.mobilenet_v2 import port_weights as port

# Reduced-precision float32 matmuls are the default on some CPUs; pin the
# precision so the feature tolerances below hold on any machine.
jax.config.update("jax_default_matmul_precision", "highest")

WEIGHTS_FILE = "mobilenetv2_bottleneck_wts.pt"
NORM_FIELDS = ("running_mean", "running_var", "weight", "bias")
DOWNLOAD = os.environ.get("PAZ_DEEP_SORT_DOWNLOAD") == "1"
SOURCES = [
    ("stem_conv", "features.0.0"),
    ("stem_bn", "features.0.1"),
    ("block_0_depthwise_conv", "features.1.conv.0"),
    ("block_0_depthwise_bn", "features.1.conv.1"),
    ("block_0_project_conv", "features.1.conv.3"),
    ("block_0_project_bn", "features.1.conv.4"),
    ("block_1_expand_conv", "features.2.conv.0"),
    ("block_1_expand_bn", "features.2.conv.1"),
    ("block_1_depthwise_conv", "features.2.conv.3"),
    ("block_1_depthwise_bn", "features.2.conv.4"),
    ("block_1_project_conv", "features.2.conv.6"),
    ("block_1_project_bn", "features.2.conv.7"),
    ("block_16_project_bn", "features.17.conv.7"),
    ("head_conv", "features.18.0"),
    ("head_bn", "features.18.1"),
]


def locate_torch_weights():
    path = os.environ.get("DEEP_SORT_MOBILENET_PT")
    return locate_packaged_weights() if path is None else path


def locate_packaged_weights():
    spec = importlib.util.find_spec("deep_sort_realtime")
    path = None
    if spec is not None:
        root = spec.submodule_search_locations[0]
        candidate = os.path.join(root, "embedder", "weights", WEIGHTS_FILE)
        path = candidate if os.path.isfile(candidate) else None
    return path


TORCH_PATH = locate_torch_weights()
REASON = "set DEEP_SORT_MOBILENET_PT or install deep-sort-realtime"
needs_weights = pytest.mark.skipif(TORCH_PATH is None, reason=REASON)


@functools.cache
def build_ported():
    model = MobileNetV2(weights=None)
    report = port.port_weights(model, TORCH_PATH)
    return model, report


@functools.cache
def load_torch_state():
    return torch.load(TORCH_PATH, map_location="cpu")


def to_numpy_state(state):
    return {key: value.numpy() for key, value in state.items()}


def build_expected_arrays(layer, state, source):
    class_name = type(layer).__name__
    if class_name == "DepthwiseConv2D":
        arrays = [np.transpose(state[f"{source}.weight"], (2, 3, 0, 1))]
    elif class_name == "Conv2D":
        arrays = [np.transpose(state[f"{source}.weight"], (2, 3, 1, 0))]
    else:
        fields = ("weight", "bias", "running_mean", "running_var")
        arrays = [state[f"{source}.{field}"] for field in fields]
    return arrays


def run_torch_reference(state, images):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.from_numpy(np.transpose(images, (0, 3, 1, 2)))
    with torch.no_grad():
        features = apply_torch_network(state, x)
    return features.numpy()


def apply_torch_network(state, x):
    x = apply_torch_unit(state, "features.0.0", "features.0.1", x, 2, True)
    for block_arg, config in enumerate(model_test.build_block_configs()):
        prefix = f"features.{block_arg + 1}.conv"
        x = apply_torch_block(state, prefix, x, config)
    x = apply_torch_unit(state, "features.18.0", "features.18.1", x, 1, True)
    return x.mean(dim=(2, 3))


def apply_torch_block(state, prefix, x, config):
    in_channels, expansion, out_channels, stride = config
    strides = [stride, 1] if expansion == 1 else [1, stride, 1]
    y = x
    for unit_arg, unit_stride in enumerate(strides):
        is_activated = unit_arg < len(strides) - 1
        keys = (f"{prefix}.{3 * unit_arg}", f"{prefix}.{3 * unit_arg + 1}")
        y = apply_torch_unit(state, *keys, y, unit_stride, is_activated)
    is_residual = stride == 1 and in_channels == out_channels
    return x + y if is_residual else y


def apply_torch_unit(state, conv_key, norm_key, x, stride, is_activated):
    weight = state[f"{conv_key}.weight"]
    groups = x.shape[1] if weight.shape[1] == 1 else 1
    padding = weight.shape[-1] // 2
    x = F.conv2d(x, weight, None, stride, padding, 1, groups)
    statistics = [state[f"{norm_key}.{field}"] for field in NORM_FIELDS]
    x = F.batch_norm(x, *statistics, False, 0.0, 1e-5)
    return F.relu6(x) if is_activated else x


def compute_relative_error(values, expected):
    return np.linalg.norm(values - expected) / np.linalg.norm(expected)


def build_images(seed):
    images = np.random.RandomState(seed).randn(2, 224, 224, 3)
    return images.astype(np.float32)


def assert_same_weights(model, other_model):
    # weighted layers only: the input layer name is not part of the contract
    weighted = [layer for layer in model.layers if layer.weights]
    assert len(weighted) == 104
    for layer in weighted:
        other_arrays = other_model.get_layer(layer.name).get_weights()
        assert len(other_arrays) == len(layer.get_weights())
        for array, other_array in zip(layer.get_weights(), other_arrays):
            assert np.array_equal(array, other_array)


def test_convert_kernel_moves_output_channels_last():
    weight = np.arange(8 * 5 * 3 * 3, dtype=np.float32).reshape(8, 5, 3, 3)
    kernel = port.convert_kernel(weight)
    assert kernel.shape == (3, 3, 5, 8)
    assert np.array_equal(kernel, np.transpose(weight, (2, 3, 1, 0)))


def test_convert_depthwise_kernel_keeps_channel_multiplier_last():
    weight = np.arange(6 * 1 * 3 * 3, dtype=np.float32).reshape(6, 1, 3, 3)
    kernel = port.convert_depthwise_kernel(weight)
    assert kernel.shape == (3, 3, 6, 1)
    assert np.array_equal(kernel, np.transpose(weight, (2, 3, 0, 1)))


def test_convert_batch_normalization_orders_keras_weights():
    fields = ("weight", "bias", "running_mean", "running_var")
    state = {"norm.num_batches_tracked": np.array(7)}
    for arg, field in enumerate(fields):
        state[f"norm.{field}"] = np.full(4, float(arg), np.float32)
    arrays = port.convert_batch_normalization(state, "norm")
    assert len(arrays) == 4
    for arg, array in enumerate(arrays):
        assert np.array_equal(array, np.full(4, float(arg), np.float32))


@pytest.mark.parametrize("layer_name, source", SOURCES)
def test_find_source_maps_layer_to_torch_module(layer_name, source):
    assert port.find_source(layer_name) == source


@needs_weights
def test_report_uses_every_key_once():
    report = build_ported()[1]
    keys = set(load_torch_state())
    assert len(keys) == 312
    assert (len(report.used), len(report.skipped)) == (260, 52)
    assert len(report.unused) == 0
    assert set(report.used).isdisjoint(report.skipped)
    assert set(report.used) | set(report.skipped) == keys
    for key in report.skipped:
        assert key.endswith(".num_batches_tracked")


@needs_weights
def test_every_keras_weight_is_assigned_once():
    model = MobileNetV2(weights=None)
    state = port.load_state(TORCH_PATH)
    assignments = port.build_assignments(model, state)
    names = [assignment.layer_name for assignment in assignments]
    keys = [key for assignment in assignments for key in assignment.keys]
    weighted = {layer.name for layer in model.layers if layer.weights}
    assert len(names) == len(set(names)) and set(names) == weighted
    assert len(keys) == len(set(keys)) == 260
    for assignment in assignments:
        layer_arrays = model.get_layer(assignment.layer_name).get_weights()
        layer_shapes = [array.shape for array in layer_arrays]
        assert [array.shape for array in assignment.arrays] == layer_shapes


@needs_weights
def test_ported_arrays_equal_torch_arrays():
    model = build_ported()[0]
    state = to_numpy_state(load_torch_state())
    for layer in model.layers:
        if layer.weights:
            source = port.find_source(layer.name)
            expected = build_expected_arrays(layer, state, source)
            arrays = layer.get_weights()
            assert len(arrays) == len(expected)
            for array, expected_array in zip(arrays, expected):
                assert array.dtype == np.float32
                assert np.array_equal(array, expected_array)


@needs_weights
def test_saved_weights_reload_bit_identical(tmp_path):
    model = build_ported()[0]
    path = str(tmp_path / "mobilenet_v2.weights.h5")
    model.save_weights(path)
    assert_same_weights(model, MobileNetV2(weights=path))


@needs_weights
def test_convert_writes_loadable_weights(tmp_path):
    path = str(tmp_path / "mobilenet_v2_paz_jax.weights.h5")
    report = port.convert(TORCH_PATH, path)
    assert (len(report.used), len(report.skipped)) == (260, 52)
    assert len(report.unused) == 0
    assert_same_weights(build_ported()[0], MobileNetV2(weights=path))


@needs_weights
def test_convert_refuses_state_with_unused_keys(tmp_path):
    state = dict(load_torch_state())
    state["features.19.weight"] = torch.zeros(1)
    torch_path = str(tmp_path / "extra_key.pt")
    torch.save(state, torch_path)
    output_path = tmp_path / "mobilenet_v2_paz_jax.weights.h5"
    with pytest.raises(ValueError):
        port.convert(torch_path, str(output_path))
    assert not output_path.exists()


@needs_weights
def test_features_match_torch_reference():
    images = build_images(0)
    features = np.asarray(build_ported()[0](images))
    expected = run_torch_reference(load_torch_state(), images)
    model_test.assert_informative(expected)
    assert features.shape == expected.shape == (2, 1280)
    assert np.allclose(features, expected, rtol=0, atol=1e-4)
    assert compute_relative_error(features, expected) <= 1e-5


@needs_weights
@pytest.mark.skipif(not DOWNLOAD, reason="set PAZ_DEEP_SORT_DOWNLOAD=1")
def test_downloaded_weights_match_local_port():
    with open(pretrained.download_weights(), "rb") as weights_file:
        digest = hashlib.sha256(weights_file.read()).hexdigest()
    assert pretrained.SHA256 is not None and digest == pretrained.SHA256
    model, ported = MobileNetV2(weights="pretrained"), build_ported()[0]
    assert_same_weights(ported, model)
    images = build_images(1)
    features = np.asarray(model(images))
    model_test.assert_informative(features)
    assert np.array_equal(features, np.asarray(ported(images)))
