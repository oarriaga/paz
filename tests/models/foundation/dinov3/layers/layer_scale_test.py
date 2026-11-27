import pytest

torch = pytest.importorskip("torch")

import numpy as np

from paz.models.foundation.dinov3.layers.layer_scale import LayerScale
from torch_layers_for_testing import PT_LayerScale


def port_weights(torch_model, keras_model):
    """Copies the gamma weight from the PyTorch model to the Keras model."""
    keras_model.gamma.assign(torch_model.gamma.detach().cpu().numpy())


@pytest.fixture(scope="module")
def params():
    """Provides common parameters for the tests and sets random seeds."""
    torch.manual_seed(0)
    np.random.seed(0)
    return {"DIM": 128, "BATCH": 4, "SEQ_LEN": 64}


def test_basic_functionality(params):
    """Tests if the Keras LayerScale output matches the PyTorch equivalent."""
    # 1. Initialize models
    torch_ls = PT_LayerScale(params["DIM"], init_values=1e-5)
    keras_ls = LayerScale(params["DIM"], init_values=1e-5)

    # 2. Prepare input tensors
    x_torch = torch.rand(params["BATCH"], params["DIM"])
    x_keras = x_torch.cpu().numpy()

    # 3. Build Keras layer and port weights for consistency
    _ = keras_ls(x_keras)
    port_weights(torch_ls, keras_ls)

    # 4. Get and compare outputs
    torch_out = torch_ls(x_torch).detach().cpu().numpy()
    keras_out = keras_ls(x_keras)

    np.testing.assert_allclose(
        torch_out, keras_out, atol=1e-7, err_msg="Basic functionality test failed."
    )


def test_non_default_initialization(params):
    """Tests the layers with a non-default initial value."""
    init_val = 0.1
    torch_ls = PT_LayerScale(params["DIM"], init_values=init_val)
    keras_ls = LayerScale(params["DIM"], init_values=init_val)

    x_torch = torch.rand(params["BATCH"], params["DIM"])
    x_keras = x_torch.cpu().numpy()

    _ = keras_ls(x_keras)
    port_weights(torch_ls, keras_ls)

    torch_out = torch_ls(x_torch).detach().cpu().numpy()
    keras_out = keras_ls(x_keras)

    np.testing.assert_allclose(
        torch_out,
        keras_out,
        atol=1e-5,
        err_msg="Non-default initialization test failed.",
    )


def test_3d_input_broadcasting(params):
    """Tests if the layer correctly broadcasts the gamma weight over a 3D input."""
    torch_ls = PT_LayerScale(params["DIM"])
    keras_ls = LayerScale(params["DIM"])

    x_torch = torch.rand(params["BATCH"], params["SEQ_LEN"], params["DIM"])
    x_keras = x_torch.cpu().numpy()

    _ = keras_ls(x_keras)
    port_weights(torch_ls, keras_ls)

    torch_out = torch_ls(x_torch).detach().cpu().numpy()
    keras_out = keras_ls(x_keras)

    np.testing.assert_allclose(
        torch_out, keras_out, atol=1e-7, err_msg="3D input broadcasting test failed."
    )


def test_keras_trainability(params):
    """Verifies that the gamma variable is the only trainable weight."""
    keras_ls = LayerScale(params["DIM"])
    _ = keras_ls(np.zeros((1, params["DIM"])))

    assert (
        len(keras_ls.trainable_weights) == 1
    ), f"Expected 1 trainable weight, found {len(keras_ls.trainable_weights)}"
    assert keras_ls.trainable_weights[0].name.startswith(
        "gamma"
    ), f"Trainable weight name should be 'gamma', but got '{keras_ls.trainable_weights[0].name}'"
