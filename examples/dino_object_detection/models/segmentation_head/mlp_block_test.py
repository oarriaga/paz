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
    MLPBlock as MLPBlock_PyTorch,
)

# Keras 3 Implementation
from examples.dino_object_detection.models.segmentation_head.mlp_block import (
    MLPBlock as MLPBlock_Keras,
)


def transfer_mlp_weights(torch_model, keras_model):
    state_dict = torch_model.state_dict()

    # 1. Norm
    keras_model.norm_in.gamma.assign(state_dict["norm_in.weight"].numpy())
    keras_model.norm_in.beta.assign(state_dict["norm_in.bias"].numpy())

    # 2. FC1 (Linear -> Dense requires transpose (Out, In) -> (In, Out))
    fc1_w = state_dict["layers.0.weight"].numpy().T
    keras_model.fc1.kernel.assign(fc1_w)
    keras_model.fc1.bias.assign(state_dict["layers.0.bias"].numpy())

    # 3. FC2
    fc2_w = state_dict["layers.2.weight"].numpy().T
    keras_model.fc2.kernel.assign(fc2_w)
    keras_model.fc2.bias.assign(state_dict["layers.2.bias"].numpy())

    # 4. Gamma
    if torch_model.gamma is not None:
        keras_model.gamma.assign(state_dict["gamma"].numpy())


def test_mlp_block_parity():
    dim = 64
    batch_size = 2

    # Init Models
    torch_model = MLPBlock_PyTorch(dim=dim, layer_scale_init_value=1e-4)
    torch_model.eval()

    keras_model = MLPBlock_Keras(dim=dim, layer_scale_init_value=1e-4)
    # Build Keras shape
    keras_model(keras.random.uniform((batch_size, 32, 32, dim)))

    # Sync Weights
    transfer_mlp_weights(torch_model, keras_model)

    # Input (NHWC format for both in this block)
    np_input = np.random.randn(batch_size, 32, 32, dim).astype(np.float32)

    # Run
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(np_input))
        torch_out = torch_out.numpy()

    keras_out = keras_model(np_input)
    if hasattr(keras_out, "numpy"):
        keras_out = keras_out.numpy()

    np.testing.assert_allclose(keras_out, torch_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
