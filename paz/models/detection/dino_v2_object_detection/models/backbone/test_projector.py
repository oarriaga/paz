import importlib.util
import os
import sys

import numpy as np
import pytest

_PT_DIR = "examples/rf-detr_original_pytorch_implementation"
_PT_ROOT = os.path.join(os.path.dirname(__file__), *([".."] * 6))
if not os.path.isdir(os.path.join(_PT_ROOT, _PT_DIR)):
    pytest.skip("RF-DETR reference unavailable", allow_module_level=True)
pytest.importorskip("torch")

import torch
import keras

from paz.models.detection.dino_v2_object_detection.models.backbone.projector import (  # noqa: E501  # fmt: skip
    MultiScaleProjector as KerasMultiScaleProjector,
    SimpleProjector as KerasSimpleProjector,
    build_conv_x,
    build_c2f,
)
from paz.models.detection.dino_v2_object_detection.models.backbone.projector_weights_porting_utils import (  # noqa: E501  # fmt: skip
    copy_ln,
    copy_weights_convx,
    copy_weights_c2f,
    port_weights_multiscale_projector,
)


def load_torch_projector():
    here = os.path.dirname(__file__)
    path = os.path.abspath(os.path.join(
        here, "../../../../../../", "examples",
        "rf-detr_original_pytorch_implementation", "rfdetr", "models",
        "backbone", "projector.py",
    ))
    spec = importlib.util.spec_from_file_location("torch_projector", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["torch_projector"] = module
    spec.loader.exec_module(module)
    return module


torch_projector = load_torch_projector()
TorchMultiScaleProjector = torch_projector.MultiScaleProjector
TorchSimpleProjector = torch_projector.SimpleProjector
TorchConvX = torch_projector.ConvX
TorchC2f = torch_projector.C2f


def set_seed():
    np.random.seed(42)
    torch.manual_seed(42)


def to_numpy(t):
    return t.detach().cpu().numpy()


def as_list(outputs):
    return outputs if isinstance(outputs, list) else [outputs]


def build_convx_model(in_ch, out_ch, kernel, stride, act, layer_norm, name):
    x = keras.Input((None, None, in_ch))
    args = (x, in_ch, out_ch, kernel, stride, 1, 1, act, layer_norm, False, name)  # fmt: skip
    return keras.Model(x, build_conv_x(*args), name="convx_model")


def build_c2f_model(in_ch, out_ch, n, shortcut, name):
    x = keras.Input((None, None, in_ch))
    args = (x, in_ch, out_ch, n, shortcut, 1, 0.5, "silu", False, False, name)
    return keras.Model(x, build_c2f(*args), name="c2f_model")


def run_torch(torch_module, x_np):
    device = next(torch_module.parameters()).device
    x_torch = torch.from_numpy(x_np.transpose(0, 3, 1, 2)).to(device)
    with torch.no_grad():
        out_torch = torch_module(x_torch)
    return to_numpy(out_torch).transpose(0, 2, 3, 1)


def test_convx_parity():
    set_seed()
    in_ch, out_ch = 32, 64
    t_mod = TorchConvX(in_ch, out_ch, kernel=3, stride=1, act="silu", layer_norm=False)  # fmt: skip
    t_mod.eval()
    k_mod = build_convx_model(in_ch, out_ch, 3, 1, "silu", False, "convx")
    copy_weights_convx(t_mod, k_mod, "convx")
    x_np = np.random.randn(1, 32, 32, in_ch).astype("float32")
    out_keras = k_mod(x_np, training=False)
    np.testing.assert_allclose(run_torch(t_mod, x_np), out_keras, rtol=1e-5, atol=1e-5)  # fmt: skip


def test_c2f_parity():
    set_seed()
    in_ch, out_ch = 64, 64
    t_mod = TorchC2f(in_ch, out_ch, n=2, shortcut=True)
    t_mod.eval()
    k_mod = build_c2f_model(in_ch, out_ch, 2, True, "c2f")
    copy_weights_c2f(t_mod, k_mod, "c2f")
    x_np = np.random.randn(1, 32, 32, in_ch).astype("float32")
    out_keras = k_mod(x_np, training=False)
    np.testing.assert_allclose(run_torch(t_mod, x_np), out_keras, rtol=1e-5, atol=1e-5)  # fmt: skip


def test_multiscale_projector_parity():
    set_seed()
    in_channels = [64, 128, 256]
    out_channels = 64
    scale_factors = [4.0, 2.0, 1.0, 0.5]
    t_mod = TorchMultiScaleProjector(in_channels, out_channels, scale_factors)
    t_mod.eval()
    input_scales = [1.0] * len(in_channels)
    k_mod = KerasMultiScaleProjector(in_channels, out_channels, scale_factors, input_scales=input_scales)  # fmt: skip
    port_weights_multiscale_projector(t_mod, k_mod)
    size = 32
    x_np = [np.random.randn(1, size, size, c).astype("float32") for c in in_channels]  # fmt: skip
    x_torch = [torch.from_numpy(x.transpose(0, 3, 1, 2)) for x in x_np]
    with torch.no_grad():
        out_torch = t_mod(x_torch)
    out_keras = as_list(k_mod(x_np))
    for o_t, o_k in zip(out_torch, out_keras):
        o_t_np = to_numpy(o_t).transpose(0, 2, 3, 1)
        np.testing.assert_allclose(o_t_np, o_k, rtol=1e-4, atol=1e-4)


def test_simple_projector_parity():
    set_seed()
    t_mod = TorchSimpleProjector(64, 64)
    t_mod.eval()
    k_mod = KerasSimpleProjector(64, 64)
    copy_weights_convx(t_mod.convx1, k_mod, "convx1")
    copy_weights_convx(t_mod.convx2, k_mod, "convx2")
    copy_ln(t_mod.ln, k_mod.get_layer("ln"))
    x_np = [np.random.randn(1, 32, 32, 64).astype("float32")]
    x_torch = [torch.from_numpy(x_np[0].transpose(0, 3, 1, 2))]
    with torch.no_grad():
        out_torch = t_mod(x_torch)
    out_keras = as_list(k_mod(x_np, training=False))
    o_t_np = to_numpy(out_torch[0]).transpose(0, 2, 3, 1)
    np.testing.assert_allclose(o_t_np, out_keras[0], rtol=1e-5, atol=1e-5)


def test_conv_transpose_parity():
    set_seed()
    in_ch, out_ch = 64, 32
    t_mod = torch.nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)  # fmt: skip
    k_mod = keras.layers.Conv2DTranspose(out_ch, kernel_size=2, strides=2, padding="valid")  # fmt: skip
    x_np = np.random.randn(1, 32, 32, in_ch).astype("float32")
    k_mod(x_np)
    w = t_mod.weight.data.cpu().numpy()
    b = t_mod.bias.data.cpu().numpy()
    k_mod.set_weights([w.transpose(2, 3, 1, 0), b])
    out_keras = k_mod(x_np, training=False)
    np.testing.assert_allclose(run_torch(t_mod, x_np), out_keras, rtol=1e-5, atol=1e-5)  # fmt: skip


def test_multiscale_projector_configurations():
    set_seed()
    configs = [
        dict(in_channels=[64, 128, 256], out_channels=64,
             scale_factors=[2.0, 1.0, 0.5], layer_norm=False),
        dict(in_channels=[128], out_channels=128,
             scale_factors=[1.0], layer_norm=False),
        dict(in_channels=[64, 128], out_channels=64,
             scale_factors=[4.0, 2.0], layer_norm=False),
    ]
    for conf in configs:
        in_channels = conf["in_channels"]
        out_channels = conf["out_channels"]
        scale_factors = conf["scale_factors"]
        layer_norm = conf["layer_norm"]
        t_mod = TorchMultiScaleProjector(in_channels, out_channels, scale_factors, layer_norm=layer_norm)  # fmt: skip
        t_mod.eval()
        input_scales = [1.0] * len(in_channels)
        k_mod = KerasMultiScaleProjector(in_channels, out_channels, scale_factors, input_scales=input_scales, layer_norm=layer_norm)  # fmt: skip
        port_weights_multiscale_projector(t_mod, k_mod)
        size = 32
        x_np = [np.random.randn(1, size, size, c).astype("float32") for c in in_channels]  # fmt: skip
        x_torch = [torch.from_numpy(x.transpose(0, 3, 1, 2)) for x in x_np]
        with torch.no_grad():
            out_torch = t_mod(x_torch)
        out_keras = as_list(k_mod(x_np, training=False))
        for o_t, o_k in zip(out_torch, out_keras):
            o_t_np = to_numpy(o_t).transpose(0, 2, 3, 1)
            np.testing.assert_allclose(o_t_np, o_k, rtol=1e-4, atol=1e-4)


def test_multiscale_projector_extra_pool():
    set_seed()
    in_channels = [64]
    out_channels = 32
    scale_factors = [2.0, 1.0, 0.5, 0.25]
    k_mod = KerasMultiScaleProjector(in_channels, out_channels, scale_factors, input_scales=[1.0])  # fmt: skip
    size = 32
    x_np = [np.random.randn(1, size, size, 64).astype("float32")]
    out_keras = as_list(k_mod(x_np, training=False))
    expected_shapes = [
        (1, 64, 64, 32),
        (1, 32, 32, 32),
        (1, 16, 16, 32),
        (1, 8, 8, 32),
    ]
    assert len(out_keras) == 4
    for out, expected in zip(out_keras, expected_shapes):
        assert out.shape == expected


if __name__ == "__main__":
    pytest.main([__file__])
