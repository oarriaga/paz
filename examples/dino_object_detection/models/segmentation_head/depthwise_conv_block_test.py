import pytest
import numpy as np
import torch
import torch.nn.functional as F
import keras
from keras import ops
import os
import sys

# -------------------------------------------------------------------------
# 0. Environment Setup
# -------------------------------------------------------------------------
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow", "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
sys.path.append(project_root)

print(f"Project Root: {project_root}")

# -------------------------------------------------------------------------
# 1. Imports
# -------------------------------------------------------------------------
# PyTorch Implementation
from examples.dino_object_detection.models.segmentation_head.torch_segmentation_head_for_testing import (
    DepthwiseConvBlock as DepthwiseConvBlock_PyTorch,
)

# Keras 3 Implementation
from examples.dino_object_detection.models.segmentation_head.depthwise_conv_block import (
    DepthwiseConvBlock as DepthwiseConvBlock_Keras,
)


def transfer_weights(torch_model, keras_model):
    """
    Transfers weights from PyTorch to Keras, handling shape permutations.
    """
    state_dict = torch_model.state_dict()

    # 1. Depthwise Conv
    dw_w = state_dict["dwconv.weight"].numpy()
    dw_w = np.transpose(dw_w, (2, 3, 1, 0))
    keras_model.dwconv.kernel.assign(dw_w)

    if "dwconv.bias" in state_dict:
        keras_model.dwconv.bias.assign(state_dict["dwconv.bias"].numpy())

    # 2. LayerNorm
    keras_model.norm.gamma.assign(state_dict["norm.weight"].numpy())
    keras_model.norm.beta.assign(state_dict["norm.bias"].numpy())

    # 3. Pointwise Conv (Linear/Dense)
    pw_w = state_dict["pwconv1.weight"].numpy()
    pw_w = np.transpose(pw_w, (1, 0))
    keras_model.pwconv1.kernel.assign(pw_w)
    keras_model.pwconv1.bias.assign(state_dict["pwconv1.bias"].numpy())

    # 4. Gamma (Layer Scale)
    if torch_model.gamma is not None:
        keras_model.gamma.assign(state_dict["gamma"].numpy())


def test_depthwise_block_parity():
    # Setup
    dim = 64
    init_val = 1e-4
    batch_size = 2

    # Initialize models
    torch_model = DepthwiseConvBlock_PyTorch(dim=dim, layer_scale_init_value=init_val)
    torch_model.eval()

    # Build Keras model (call once to create weights)
    keras_model = DepthwiseConvBlock_Keras(dim=dim, layer_scale_init_value=init_val)
    dummy_input_keras = keras.random.uniform((batch_size, dim, 32, 32))
    keras_model(dummy_input_keras)

    # Sync Weights
    transfer_weights(torch_model, keras_model)

    # Generate Identical Input
    # NCHW Format for both (since your Keras block expects channels_first)
    np_input = np.random.randn(batch_size, dim, 32, 32).astype(np.float32)

    # Forward Pass PyTorch
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(np_input))
        torch_out = torch_out.numpy()

    # Forward Pass Keras
    keras_out = keras_model(np_input)
    if hasattr(keras_out, "numpy"):
        keras_out = keras_out.numpy()
    else:  # Handle JAX/Numpy backends
        keras_out = np.array(keras_out)

    # Comparison
    diff = np.abs(torch_out - keras_out)
    print(f"\nMax Difference: {diff.max()}")

    np.testing.assert_allclose(
        keras_out,
        torch_out,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Output mismatch between PyTorch and Keras implementation",
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
