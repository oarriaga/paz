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
    point_sample as point_sample_PyTorch,
    get_uncertain_point_coords_with_randomness as get_uncertain_point_coords_with_randomness_PyTorch,
)

# Keras 3 Implementation
from examples.dino_object_detection.models.segmentation_head.utils import (
    point_sample as point_sample_Keras,
    get_uncertain_point_coords_with_randomness as get_uncertain_point_coords_with_randomness_Keras,
)


import pytest
import numpy as np
import torch
import keras
from keras import ops


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def to_numpy(x):
    """Convert Torch tensors or Keras/JAX arrays to a standard Numpy array."""
    if hasattr(x, "detach"):  # PyTorch
        return x.detach().cpu().numpy()
    return np.array(x)  # Keras / JAX / TensorFlow


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


def test_point_sample_exact_match():
    """
    Verifies that Keras point_sample produces the exact same interpolated
    features as the PyTorch implementation for fixed inputs.
    """
    # 1. Setup deterministic input data
    N, C, H, W = 2, 4, 32, 32
    num_points = 100

    # Random floats [-1, 1]
    np_input = np.random.randn(N, C, H, W).astype("float32")
    # Random coords [0, 1]
    np_points = np.random.rand(N, num_points, 2).astype("float32")

    # 2. Run PyTorch
    th_input = torch.tensor(np_input)
    th_points = torch.tensor(np_points)
    th_out = point_sample_PyTorch(th_input, th_points, align_corners=False)

    # 3. Run Keras
    ks_input = ops.convert_to_tensor(np_input)
    ks_points = ops.convert_to_tensor(np_points)
    ks_out = point_sample_Keras(ks_input, ks_points, align_corners=False)

    # 4. Assert Numerical Equivalence
    # We use a small tolerance (atol=1e-5) for floating point differences between backends
    np.testing.assert_allclose(
        to_numpy(ks_out),
        to_numpy(th_out),
        atol=1e-5,
        err_msg="Mismatch between PyTorch and Keras point_sample results",
    )


def test_uncertainty_sampling_structure():
    """
    Verifies that the uncertainty sampling produces valid shapes and bounds.
    Note: We cannot check for value equality because both functions use
    internal random number generators (torch.rand vs ops.random) that
    won't sync across frameworks.
    """
    # 1. Setup
    N, C, H, W = 2, 1, 64, 64
    num_points = 112
    np_logits = np.random.randn(N, C, H, W).astype("float32")

    # Define simple uncertainty functions for both frameworks
    func_torch = lambda x: -torch.abs(x)
    func_keras = lambda x: -ops.abs(x)

    # 2. Run PyTorch
    th_out = get_uncertain_point_coords_with_randomness_PyTorch(
        torch.tensor(np_logits), func_torch, num_points=num_points
    )

    # 3. Run Keras
    ks_out = get_uncertain_point_coords_with_randomness_Keras(
        ops.convert_to_tensor(np_logits), func_keras, num_points=num_points
    )

    # 4. Assertions
    # Check Shapes: Should be (N, num_points, 2)
    assert tuple(th_out.shape) == (N, num_points, 2)
    assert tuple(ks_out.shape) == (N, num_points, 2)

    # Check Bounds: Coordinates must be in [0, 1]
    th_vals = to_numpy(th_out)
    ks_vals = to_numpy(ks_out)

    assert th_vals.min() >= 0.0 and th_vals.max() <= 1.0, "PyTorch output out of bounds"
    assert ks_vals.min() >= 0.0 and ks_vals.max() <= 1.0, "Keras output out of bounds"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
