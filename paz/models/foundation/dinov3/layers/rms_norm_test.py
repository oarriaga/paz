import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import numpy as np
import jax
import keras
import torch
from keras import ops


# ==============================================================================
# Keras Layer Implementation
# ==============================================================================
from paz.models.foundation.dinov3.layers.rms_norm import RMSNorm

# ==============================================================================
# PyTorch Reference Implementation (unchangeable)
# ==============================================================================
from paz.models.foundation.dinov3.layers.torch_layers_for_testing import PT_RMSNorm

# ==============================================================================
# Test Suite
# ==============================================================================


@pytest.fixture
def params():
    """Provides common parameters for tests."""
    return {
        "dim": 128,
        "batch_size": 4,
        "seq_len": 64,
        "epsilon": 1e-5,
    }


def run_and_compare(keras_model, torch_model, input_np, atol=1e-5):
    """Helper function to execute models and assert closeness."""
    # Keras/JAX execution
    output_keras = keras_model(input_np)
    output_keras_np = np.array(output_keras)

    # PyTorch execution
    input_torch = torch.from_numpy(input_np)
    output_torch = torch_model(input_torch)
    output_torch_np = output_torch.detach().numpy()

    # Compare shapes and values
    assert output_keras_np.shape == output_torch_np.shape
    np.testing.assert_allclose(output_keras_np, output_torch_np, atol=atol)
    return output_keras_np, output_torch_np


# === Main Equivalence Tests ===


def test_basic_equivalence(params):
    """Tests if outputs are identical for a standard float32 input."""
    shape = (params["batch_size"], params["seq_len"], params["dim"])
    input_np = np.random.rand(*shape).astype("float32")

    keras_model = RMSNorm(epsilon=params["epsilon"])
    torch_model = PT_RMSNorm(dim=params["dim"], eps=params["epsilon"])

    run_and_compare(keras_model, torch_model, input_np)


@pytest.mark.parametrize(
    "shape_params",
    [
        (128,),
        (16, 64),
        (4, 10, 32),
        (2, 8, 8, 16),
    ],
)
def test_different_shapes(shape_params):
    """Tests equivalence across various input tensor shapes."""
    dim = shape_params[-1]
    input_np = np.random.rand(*shape_params).astype("float32")
    keras_model = RMSNorm()
    torch_model = PT_RMSNorm(dim=dim)
    run_and_compare(keras_model, torch_model, input_np)


def test_gradients(params):
    """Ensures the backward pass (gradients) is also equivalent."""
    shape = (params["batch_size"], params["dim"])
    input_np = np.random.rand(*shape).astype("float32")
    downstream_grad_np = np.random.rand(*shape).astype("float32")

    # PyTorch Gradients
    torch_model = PT_RMSNorm(dim=params["dim"])
    torch_model.train()
    input_torch = torch.from_numpy(input_np)
    output_torch = torch_model(input_torch)
    output_torch.backward(gradient=torch.from_numpy(downstream_grad_np))
    grad_torch = torch_model.weight.grad.numpy()

    # Keras/JAX Gradients
    keras_model = RMSNorm(epsilon=params["epsilon"])
    keras_model.build(input_np.shape)

    def get_loss(trainable_weights):
        temp_keras_model = RMSNorm(epsilon=params["epsilon"])
        temp_keras_model.build(input_np.shape)
        temp_keras_model.set_weights(trainable_weights)
        output = temp_keras_model(input_np)
        return ops.sum(output * downstream_grad_np)

    grad_fn = jax.grad(get_loss)
    grad_jax_list = grad_fn(keras_model.trainable_weights)
    grad_jax = grad_jax_list[0]

    np.testing.assert_allclose(np.array(grad_jax), grad_torch, atol=1e-5)


# === Edge Case Tests ===


def test_zero_input(params):
    """Tests that an all-zero input produces an all-zero output."""
    shape = (params["batch_size"], params["dim"])
    input_np = np.zeros(shape, dtype="float32")
    keras_model = RMSNorm(epsilon=params["epsilon"])
    torch_model = PT_RMSNorm(dim=params["dim"], eps=params["epsilon"])
    output_keras, _ = run_and_compare(keras_model, torch_model, input_np)
    assert not np.any(output_keras)


def test_nan_propagation(params):
    """Tests if NaN in the input correctly propagates to the output."""
    shape = (params["batch_size"], params["dim"])
    input_np = np.random.rand(*shape).astype("float32")
    input_np[0, 5] = np.nan

    keras_model = RMSNorm(epsilon=params["epsilon"])
    torch_model = PT_RMSNorm(dim=params["dim"], eps=params["epsilon"])

    output_keras = np.array(keras_model(input_np))
    output_torch = torch_model(torch.from_numpy(input_np)).detach().numpy()

    assert np.isnan(output_keras[0, 5])
    assert np.isnan(output_torch[0, 5])


# === Pre-trained Weights Tests ===

# Define paths to local model files. Update these to match your system.
DINO_REPO_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
DINO_WEIGHT_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
dinov3_files_exist = os.path.isdir(DINO_REPO_PATH) and os.path.isfile(DINO_WEIGHT_PATH)


@pytest.mark.skipif(
    not dinov3_files_exist, reason="DINOv3 local model/weight files not found."
)
def test_dinov3_real_pretrained_weights(params):
    """
    Tests equivalence by loading ACTUAL pre-trained weights from a DINOv3
    model file and comparing outputs.
    """
    # 1. Load the pre-trained DINOv3 model from local source
    dinov3_model = torch.hub.load(
        DINO_REPO_PATH, "dinov3_vits16", source="local", weights=DINO_WEIGHT_PATH
    )
    dinov3_model.eval()

    # 2. Extract the real LayerNorm layer and its parameters
    torch_pretrained_norm = dinov3_model.blocks[0].norm1
    dino_dim = torch_pretrained_norm.weight.shape[0]
    dino_eps = torch_pretrained_norm.eps

    # 3. Use the built-in Keras LayerNormalization layer
    keras_model = keras.layers.LayerNormalization(
        epsilon=dino_eps,
        # Ensure parameter names match for weight loading
        gamma_initializer="ones",
        beta_initializer="zeros",
    )

    # 4. Create matching random input data
    shape = (params["batch_size"], params["seq_len"], dino_dim)
    input_np = np.random.rand(*shape).astype("float32")

    # 5. Build Keras layer and port the pre-trained weights
    keras_model.build(input_np.shape)
    pretrained_weight = torch_pretrained_norm.weight.detach().cpu().numpy()
    pretrained_bias = torch_pretrained_norm.bias.detach().cpu().numpy()
    keras_model.set_weights([pretrained_weight, pretrained_bias])

    # 6. Run forward pass and compare outputs
    run_and_compare(keras_model, torch_pretrained_norm, input_np, atol=1e-5)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
