import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import keras
from keras import ops
import os
import sys

# -------------------------------------------------------------------------
# 0. Setup & Imports
# -------------------------------------------------------------------------
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Add project root to path for imports
script_path = os.path.abspath(__file__)
project_root = os.path.abspath(
    os.path.join(os.path.dirname(script_path), "..", "..", "..", "..")
)
sys.path.append(project_root)


# -------------------------------------------------------------------------
# 1. Imports
# -------------------------------------------------------------------------
# PyTorch Implementation

# Keras 3 Implementation
from examples.dino_object_detection.models.segmentation_head.segmentation_head import (
    SegmentationHead as SegmentationHead_Keras,
)


# -------------------------------------------------------------------------
# 1. PyTorch Reference Models
# -------------------------------------------------------------------------

from examples.dino_object_detection.models.segmentation_head.torch_segmentation_head_for_testing import (
    SegmentationHead as SegmentationHead_PyTorch,
)


# -------------------------------------------------------------------------
# 2. Comparison & Transfer Helpers
# -------------------------------------------------------------------------
def assert_tensors_equal(keras_t, torch_t, tol=1e-5, msg="Mismatch"):
    k_np = keras_t.numpy() if hasattr(keras_t, "numpy") else np.array(keras_t)
    t_np = torch_t.detach().cpu().numpy()
    np.testing.assert_allclose(k_np, t_np, rtol=tol, atol=tol, err_msg=msg)


def transfer_weights(torch_model, keras_model):
    """Transfers weights from PyTorch to Keras, handling shape permutations."""
    # 1. Blocks
    for pt_block, k_block in zip(torch_model.blocks, keras_model.blocks):
        # Depthwise
        dw_w = pt_block.dwconv.weight.detach().numpy()
        k_block.dwconv.kernel.assign(np.transpose(dw_w, (2, 3, 1, 0)))  # (H, W, 1, C)
        if pt_block.dwconv.bias is not None:
            k_block.dwconv.bias.assign(pt_block.dwconv.bias.detach().numpy())

        # Norm & Pointwise
        k_block.norm.gamma.assign(pt_block.norm.weight.detach().numpy())
        k_block.norm.beta.assign(pt_block.norm.bias.detach().numpy())
        k_block.pwconv1.kernel.assign(
            pt_block.pwconv1.weight.detach().numpy().T
        )  # (In, Out)
        k_block.pwconv1.bias.assign(pt_block.pwconv1.bias.detach().numpy())
        if pt_block.gamma is not None:
            k_block.gamma.assign(pt_block.gamma.detach().numpy())

    # 2. Spatial Projector (Conv2d 1x1 -> Dense)
    if hasattr(torch_model.spatial_features_proj, "weight"):
        w = torch_model.spatial_features_proj.weight.detach().numpy()  # (Out, In, 1, 1)
        w = w.reshape(w.shape[0], w.shape[1]).T  # (In, Out)
        keras_model.spatial_features_proj.kernel.assign(w)
        keras_model.spatial_features_proj.bias.assign(
            torch_model.spatial_features_proj.bias.detach().numpy()
        )

    # 3. Query Projector (Linear -> Dense)
    if hasattr(torch_model.query_features_proj, "weight"):
        keras_model.query_features_proj.kernel.assign(
            torch_model.query_features_proj.weight.detach().numpy().T
        )
        keras_model.query_features_proj.bias.assign(
            torch_model.query_features_proj.bias.detach().numpy()
        )

    # 4. Query MLP
    qm_pt, qm_k = torch_model.query_features_block, keras_model.query_features_block
    qm_k.norm_in.gamma.assign(qm_pt.norm_in.weight.detach().numpy())
    qm_k.norm_in.beta.assign(qm_pt.norm_in.bias.detach().numpy())
    qm_k.fc1.kernel.assign(qm_pt.layers[0].weight.detach().numpy().T)
    qm_k.fc1.bias.assign(qm_pt.layers[0].bias.detach().numpy())
    qm_k.fc2.kernel.assign(qm_pt.layers[2].weight.detach().numpy().T)
    qm_k.fc2.bias.assign(qm_pt.layers[2].bias.detach().numpy())
    if qm_pt.gamma is not None:
        qm_k.gamma.assign(qm_pt.gamma.detach().numpy())

    # 5. Bias
    keras_model.bias.assign(torch_model.bias.detach().numpy())


# -------------------------------------------------------------------------
# 3. Fixtures (Setup)
# -------------------------------------------------------------------------
@pytest.fixture
def test_config():
    return {
        "B": 2,
        "C": 64,
        "H": 32,
        "W": 32,
        "queries": 5,
        "blocks": 2,
        "bottleneck": 2,
        "ds": 4,
    }


@pytest.fixture
def input_data(test_config):
    np.random.seed(42)
    B, C, H, W = test_config["B"], test_config["C"], test_config["H"], test_config["W"]
    spatial = np.random.randn(B, C, H, W).astype(np.float32)
    queries = [
        np.random.randn(B, test_config["queries"], C).astype(np.float32)
        for _ in range(test_config["blocks"])
    ]

    img_size = (H * test_config["ds"], W * test_config["ds"])
    return {
        "np_spatial": spatial,
        "np_queries": queries,
        "img_size": img_size,
        "pt_spatial": torch.from_numpy(spatial),
        "pt_queries": [torch.from_numpy(q) for q in queries],
    }


@pytest.fixture
def models(test_config, input_data):
    # Initialize PyTorch
    pt_model = SegmentationHead_PyTorch(
        test_config["C"],
        test_config["blocks"],
        test_config["bottleneck"],
        test_config["ds"],
    )
    pt_model.eval()

    # Initialize Keras
    k_model = SegmentationHead_Keras(
        test_config["C"],
        test_config["blocks"],
        test_config["bottleneck"],
        test_config["ds"],
    )
    # Build Keras model
    k_model(
        input_data["np_spatial"],
        input_data["np_queries"],
        image_size=input_data["img_size"],
    )

    # Sync
    transfer_weights(pt_model, k_model)
    return pt_model, k_model


# -------------------------------------------------------------------------
# 4. Tests
# -------------------------------------------------------------------------
def test_resizing_logic(input_data, test_config):
    """Verifies that Keras resizing (manual transpose) matches PyTorch interpolate."""
    ds = test_config["ds"]
    target_size = (input_data["img_size"][0] // ds, input_data["img_size"][1] // ds)

    # PyTorch
    pt_out = F.interpolate(
        input_data["pt_spatial"], size=target_size, mode="bilinear", align_corners=False
    )

    # Keras (Manual Logic from Call)
    k_in = ops.transpose(input_data["np_spatial"], (0, 2, 3, 1))
    k_out = ops.image.resize(k_in, target_size, interpolation="bilinear")
    k_out = ops.transpose(k_out, (0, 3, 1, 2))

    assert_tensors_equal(k_out, pt_out, atol=1e-5, msg="Resizing mismatch")


def test_single_block_parity(models, input_data):
    """Verifies the output of the first DepthwiseConvBlock."""
    pt_model, k_model = models

    # Run on identical random input (not real flow, just block logic)
    x = np.random.randn(*input_data["np_spatial"].shape).astype(np.float32)

    pt_out = pt_model.blocks[0](torch.from_numpy(x))
    k_out = k_model.blocks[0](x)

    assert_tensors_equal(k_out, pt_out, atol=1e-5, msg="Depthwise Block mismatch")


def test_spatial_projection(models, input_data):
    """Verifies the spatial feature projection (Dense vs 1x1 Conv)."""
    pt_model, k_model = models
    x = np.random.randn(*input_data["np_spatial"].shape).astype(np.float32)  # (N,C,H,W)

    pt_out = pt_model.spatial_features_proj(torch.from_numpy(x))

    # Keras Logic (N,C,H,W -> N,H,W,C -> Dense -> N,C,H,W)
    k_in = ops.transpose(x, (0, 2, 3, 1))
    k_out = k_model.spatial_features_proj(k_in)
    k_out = ops.transpose(k_out, (0, 3, 1, 2))

    assert_tensors_equal(k_out, pt_out, atol=1e-5, msg="Spatial Projection mismatch")


def test_query_projection(models, input_data):
    """Verifies the MLP block and query projection sequence."""
    pt_model, k_model = models
    q = input_data["np_queries"][0]

    # PyTorch
    pt_q = torch.from_numpy(q)
    pt_out = pt_model.query_features_proj(pt_model.query_features_block(pt_q))

    # Keras
    k_out = k_model.query_features_proj(k_model.query_features_block(q))

    assert_tensors_equal(k_out, pt_out, atol=1e-5, msg="Query Projection mismatch")


def test_full_forward_pass(models, input_data):
    """Verifies the complete output of the SegmentationHead."""
    pt_model, k_model = models

    # PyTorch Forward
    with torch.no_grad():
        pt_logits = pt_model(
            input_data["pt_spatial"],
            input_data["pt_queries"],
            image_size=input_data["img_size"],
        )

    # Keras Forward
    k_logits = k_model(
        input_data["np_spatial"],
        input_data["np_queries"],
        image_size=input_data["img_size"],
    )

    assert len(pt_logits) == len(k_logits)
    for i, (pt, k) in enumerate(zip(pt_logits, k_logits)):
        assert_tensors_equal(k, pt, atol=2e-5, msg=f"Final Logit mismatch at index {i}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
