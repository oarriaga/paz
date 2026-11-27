import pytest

torch = pytest.importorskip("torch")

import numpy as np
import jax
from keras import ops

from paz.models.foundation.dinov3.layers.rms_norm import RMSNorm
from torch_layers_for_testing import PT_RMSNorm


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
