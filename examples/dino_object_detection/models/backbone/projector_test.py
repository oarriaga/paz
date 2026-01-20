import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(project_root)

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import keras
from keras import layers

from keras import ops


# -------------------------------------------------------------------------
# PyTorch Implementation (Imports)
# -------------------------------------------------------------------------
from examples.dino_object_detection.models.backbone.torch_projector_for_testing import (
    LayerNorm as PTLayerNorm,
    ConvX as PTConvX,
    Bottleneck as PTBottleneck,
    C2f as PTC2f,
    SimpleProjector as PTSimpleProjector,
    MultiScaleProjector as PTMultiScaleProjector,
)

# -------------------------------------------------------------------------
# Keras 3 Implementation (Imports)
# -------------------------------------------------------------------------
from examples.dino_object_detection.models.backbone.projector import (
    LayerNorm as KerasLayerNorm,
    ConvX as KerasConvX,
    Bottleneck as KerasBottleneck,
    C2f as KerasC2f,
    SimpleProjector as KerasSimpleProjector,
    MultiScaleProjector as KerasMultiScaleProjector,
)


# --- Helper Functions ---
def get_data(shape=(2, 64, 32, 32)):
    np_data = np.random.randn(*shape).astype(np.float32)
    return torch.from_numpy(np_data), keras.ops.convert_to_tensor(np_data)


def build_layer(layer, input_shape):
    _ = layer(keras.ops.zeros(input_shape))


def copy_conv(pt_layer, kr_layer):
    """Copies weights robustly for BN or LN."""
    pt_w = pt_layer.conv.weight.detach().numpy()
    kr_layer.conv.kernel.assign(pt_w.transpose(2, 3, 1, 0))

    if hasattr(pt_layer, "bn") and pt_layer.bn is not None:
        pt_w = pt_layer.bn.weight.detach().numpy()
        if hasattr(pt_layer.bn, "bias") and pt_layer.bn.bias is not None:
            pt_b = pt_layer.bn.bias.detach().numpy()
        else:
            pt_b = np.zeros_like(pt_w)

        if hasattr(kr_layer.bn, "gamma"):
            kr_layer.bn.gamma.assign(pt_w)
            kr_layer.bn.beta.assign(pt_b)
            if hasattr(pt_layer.bn, "running_mean"):
                kr_layer.bn.moving_mean.assign(
                    pt_layer.bn.running_mean.detach().numpy()
                )
                kr_layer.bn.moving_variance.assign(
                    pt_layer.bn.running_var.detach().numpy()
                )
        else:
            kr_layer.bn.weight.assign(pt_w)
            kr_layer.bn.bias.assign(pt_b)


def copy_bottleneck(pt_module, kr_module):
    copy_conv(pt_module.cv1, kr_module.cv1)
    copy_conv(pt_module.cv2, kr_module.cv2)


def copy_c2f(pt_module, kr_module):
    copy_conv(pt_module.cv1, kr_module.cv1)
    copy_conv(pt_module.cv2, kr_module.cv2)
    for pt_b, kr_b in zip(pt_module.m, kr_module.bottlenecks):
        copy_bottleneck(pt_b, kr_b)


def copy_multiscale_robust(pt_module, kr_module):
    for i, pt_stage_list in enumerate(pt_module.stages_sampling):
        for j, pt_seq in enumerate(pt_stage_list):
            kr_seq = kr_module.stages_sampling[i][j]
            pt_layers = list(pt_seq.children())
            kr_layers = kr_seq.layers

            if len(kr_layers) == 1 and "Identity" in str(type(kr_layers[0])):
                continue

            for pt_l, kr_l in zip(pt_layers, kr_layers):
                if isinstance(pt_l, torch.nn.ConvTranspose2d):
                    w = pt_l.weight.detach().numpy()
                    w = w.transpose(2, 3, 1, 0)

                    if hasattr(kr_l, "kernel"):
                        kr_l.kernel.assign(w)
                    else:
                        kr_l.weights[0].assign(w)

                    if pt_l.bias is not None:
                        target = kr_l.bias if hasattr(kr_l, "bias") else kr_l.weights[1]
                        target.assign(pt_l.bias.detach().numpy())

                elif isinstance(pt_l, torch.nn.Conv2d):
                    w = pt_l.weight.detach().numpy().transpose(2, 3, 1, 0)
                    if hasattr(kr_l, "kernel"):
                        kr_l.kernel.assign(w)
                    else:
                        kr_l.weights[0].assign(w)
                    if pt_l.bias is not None:
                        target = kr_l.bias if hasattr(kr_l, "bias") else kr_l.weights[1]
                        target.assign(pt_l.bias.detach().numpy())

                elif isinstance(pt_l, PTConvX):
                    copy_conv(pt_l, kr_l)

                elif isinstance(pt_l, PTLayerNorm):
                    kr_l.weight.assign(pt_l.weight.detach().numpy())
                    kr_l.bias.assign(pt_l.bias.detach().numpy())

    for pt_stage, kr_stage in zip(pt_module.stages, kr_module.stages):
        copy_c2f(pt_stage[0], kr_stage.layers[0])
        ln_pt, ln_kr = pt_stage[1], kr_stage.layers[1]
        ln_kr.weight.assign(ln_pt.weight.detach().numpy())
        ln_kr.bias.assign(ln_pt.bias.detach().numpy())


# --- Tests ---


def test_convx():
    pt_mod = PTConvX(32, 64, 3, 2).eval()
    kr_mod = KerasConvX(32, 64, 3, 2)
    x_pt, x_kr = get_data((2, 32, 16, 16))
    build_layer(kr_mod, x_kr.shape)
    copy_conv(pt_mod, kr_mod)
    np.testing.assert_allclose(
        pt_mod(x_pt).detach().numpy(), np.array(kr_mod(x_kr, training=False)), atol=1e-4
    )


def test_bottleneck():
    pt_mod = PTBottleneck(64, 64).eval()
    kr_mod = KerasBottleneck(64, 64)
    x_pt, x_kr = get_data((2, 64, 32, 32))
    build_layer(kr_mod, x_kr.shape)
    copy_bottleneck(pt_mod, kr_mod)
    np.testing.assert_allclose(
        pt_mod(x_pt).detach().numpy(), np.array(kr_mod(x_kr, training=False)), atol=1e-4
    )


def test_c2f():
    pt_mod = PTC2f(64, 64, n=2, shortcut=True).eval()
    kr_mod = KerasC2f(64, 64, n=2, shortcut=True)
    x_pt, x_kr = get_data((2, 64, 32, 32))
    build_layer(kr_mod, x_kr.shape)
    copy_c2f(pt_mod, kr_mod)
    np.testing.assert_allclose(
        pt_mod(x_pt).detach().numpy(), np.array(kr_mod(x_kr, training=False)), atol=1e-4
    )


def test_simple_projector():
    pt_mod = PTSimpleProjector(64, 128).eval()
    kr_mod = KerasSimpleProjector(64, 128)
    x_np = np.random.randn(2, 64, 32, 32).astype(np.float32)
    x_kr = [keras.ops.convert_to_tensor(x_np)]
    _ = kr_mod(x_kr)
    copy_conv(pt_mod.convx1, kr_mod.convx1)
    copy_conv(pt_mod.convx2, kr_mod.convx2)
    kr_mod.ln.weight.assign(pt_mod.ln.weight.detach().numpy())
    kr_mod.ln.bias.assign(pt_mod.ln.bias.detach().numpy())

    np.testing.assert_allclose(
        pt_mod([torch.from_numpy(x_np)])[0].detach().numpy(),
        np.array(kr_mod(x_kr)[0]),
        atol=1e-4,
    )


def test_layer_norm_variance_correction():
    """Verify Keras LayerNorm matches PyTorch LayerNorm (biased variance)."""
    normalized_shape = 32
    pt_ln = PTLayerNorm(normalized_shape, eps=1e-6).eval()
    kr_ln = KerasLayerNorm(normalized_shape, eps=1e-6)

    # Input (N, C, H, W)
    x_pt, x_kr = get_data((2, 32, 10, 10))
    build_layer(kr_ln, x_kr.shape)

    kr_ln.weight.assign(pt_ln.weight.detach().numpy())
    kr_ln.bias.assign(pt_ln.bias.detach().numpy())

    out_pt = pt_ln(x_pt).detach().numpy()
    out_kr = keras.ops.convert_to_numpy(kr_ln(x_kr))

    # Needs high precision check for variance correction
    np.testing.assert_allclose(out_pt, out_kr, rtol=1e-5, atol=1e-6)


def test_multiscale_projector():
    in_channels = [32, 64, 128]
    out_channels = 64
    scale_factors = [1.0, 2.0, 4.0]

    pt_mod = PTMultiScaleProjector(in_channels, out_channels, scale_factors).eval()
    kr_mod = KerasMultiScaleProjector(in_channels, out_channels, scale_factors)

    inputs_np = [
        np.random.randn(1, c, 32, 32).astype(np.float32)
        for i, c in enumerate(in_channels)
    ]
    x_pt = [torch.from_numpy(x) for x in inputs_np]
    x_kr = [keras.ops.convert_to_tensor(x) for x in inputs_np]

    _ = kr_mod(x_kr)
    copy_multiscale_robust(pt_mod, kr_mod)

    with torch.no_grad():
        y_pt_list = pt_mod(x_pt)
    y_kr_list = kr_mod(x_kr, training=False)

    for y_p, y_k in zip(y_pt_list, y_kr_list):
        np.testing.assert_allclose(
            y_p.detach().numpy(), np.array(y_k), rtol=1e-3, atol=1e-3
        )


def test_multiscale_extra_pool():
    """Test scale factor 0.25 which triggers extra pooling."""
    in_channels = [32]
    out_channels = 16
    scale_factors = [1.0, 0.25]  # 0.25 triggers extra pool

    kr_mod = KerasMultiScaleProjector(in_channels, out_channels, scale_factors)
    x_kr = [keras.ops.zeros((1, 32, 32, 32))]

    out = kr_mod(x_kr)

    assert len(out) == 2
    assert out[0].shape == (1, 16, 32, 32)
    assert out[1].shape == (1, 16, 16, 16)  # Pooled


def test_multiscale_force_drop():
    """Test force_drop_last_n_features."""
    in_channels = [32, 32, 32]
    out_channels = 16
    scale_factors = [1.0]

    kr_mod = KerasMultiScaleProjector(
        in_channels, out_channels, scale_factors, force_drop_last_n_features=1
    )

    x_in = [np.random.randn(1, 32, 16, 16).astype(np.float32) for _ in range(3)]
    x_kr = [keras.ops.convert_to_tensor(x) for x in x_in]

    out = kr_mod(x_kr)
    assert len(out) == 1
    assert out[0].shape == (1, 16, 16, 16)


def test_multiscale_survival_prob():
    """Test stochastic depth behaves (output changes or zeros appear)."""
    in_channels = [32, 32]
    out_channels = 16
    scale_factors = [1.0]

    kr_mod = KerasMultiScaleProjector(
        in_channels, out_channels, scale_factors, survival_prob=0.0
    )

    x_kr = [keras.ops.ones((1, 32, 16, 16)) for _ in range(2)]

    # Run in inference mode (no drop)
    out_eval = kr_mod(x_kr, training=False)

    # Run in training mode (force drop of x[1])
    out_train = kr_mod(x_kr, training=True)

    # They must differ because x[1] contributes to the result
    assert not np.allclose(out_eval[0], out_train[0])


def load_pretrained_projector_weights(pt_model, weights_path):
    # Use your specific path
    weights_path = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\rf-detr-base-coco.pth"

    print(f"Loading {weights_path}...")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Extract state_dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Print all unique prefixes to find the projector
    print("\n--- Top Level Prefixes ---")
    prefixes = set(k.split(".")[0] for k in state_dict.keys())
    print(prefixes)

    # Search specifically for projection-related keys
    print("\n--- Potential Projector Keys ---")
    proj_keys = [
        k for k in state_dict.keys() if "proj" in k or "neck" in k or "fpn" in k
    ]
    for k in proj_keys[:20]:  # Print first 20 matches
        print(k)

    if not os.path.exists(weights_path):
        pytest.skip(f"Weights file not found at: {weights_path}")

    print(f"Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    prefix = "input_proj."  # Default

    candidates = [k for k in state_dict.keys() if "stages.0" in k and "weight" in k]
    if candidates:
        common_prefix = candidates[0].split("stages.0")[0]
        print(f"Auto-detected prefix: '{common_prefix}'")
        prefix = common_prefix
    else:
        print("Could not auto-detect prefix. Using default 'input_proj.'")

    model_state_dict = pt_model.state_dict()
    final_state_dict = {}

    matched_count = 0
    for k, v in state_dict.items():
        if k.startswith(prefix):
            local_key = k[len(prefix) :]

            if local_key in model_state_dict:
                if model_state_dict[local_key].shape == v.shape:
                    final_state_dict[local_key] = v
                    matched_count += 1
                else:
                    print(
                        f"Shape mismatch for {local_key}: PT={v.shape} vs Model={model_state_dict[local_key].shape}"
                    )

    if matched_count == 0:
        pytest.fail(
            f"No keys matched! Prefix used: '{prefix}'. Run inspection script to verify keys."
        )

    pt_model.load_state_dict(final_state_dict, strict=False)
    print(f"Successfully loaded {matched_count} keys.")


def test_multiscale_projector_real_weights():
    """
    Test MultiScaleProjector parity using real pretrained weights from disk.
    """
    # 1. Configuration matching the pretrained model
    in_channels = [512, 1024, 2048]
    out_channels = 256
    scale_factors = [1.0, 2.0, 4.0]

    weights_path = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\rf-detr-base-coco.pth"

    # 2. Instantiate Models
    pt_mod = PTMultiScaleProjector(in_channels, out_channels, scale_factors).eval()
    kr_mod = KerasMultiScaleProjector(in_channels, out_channels, scale_factors)

    # 3. Load Real Weights into PyTorch
    load_pretrained_projector_weights(pt_mod, weights_path)

    # 4. Generate Random Input Data
    inputs_np = [np.random.randn(1, c, 32, 32).astype(np.float32) for c in in_channels]
    x_pt = [torch.from_numpy(x) for x in inputs_np]
    x_kr = [keras.ops.convert_to_tensor(x) for x in inputs_np]

    # 5. Build Keras Model & Transfer Weights
    _ = kr_mod(x_kr, training=False)

    # Use your existing robust copy function
    copy_multiscale_robust(pt_mod, kr_mod)

    # 6. Run Inference & Compare
    with torch.no_grad():
        y_pt_list = pt_mod(x_pt)

    y_kr_list = kr_mod(x_kr, training=False)

    print("\nVerifying Parity with Real Weights:")
    for i, (y_p, y_k) in enumerate(zip(y_pt_list, y_kr_list)):
        y_p_np = y_p.detach().numpy()
        y_k_np = np.array(y_k)

        diff = np.abs(y_p_np - y_k_np)
        mae = np.mean(diff)
        max_diff = np.max(diff)

        print(f"Scale {i}: MAE={mae:.6f}, MaxDiff={max_diff:.6f}")

        np.testing.assert_allclose(
            y_p_np, y_k_np, rtol=1e-5, atol=1e-5, err_msg=f"Mismatch at Scale {i}"
        )


def test_odd_shapes():
    """Ensure padding logic holds for odd spatial dimensions."""
    pt_mod = PTConvX(32, 32, 3, stride=2).eval()
    kr_mod = KerasConvX(32, 32, 3, stride=2)

    x_pt, x_kr = get_data((1, 32, 33, 33))

    build_layer(kr_mod, x_kr.shape)
    copy_conv(pt_mod, kr_mod)

    np.testing.assert_allclose(
        pt_mod(x_pt).detach().numpy(), np.array(kr_mod(x_kr, training=False)), atol=1e-4
    )


if __name__ == "__main__":
    pytest.main([__file__])
