import numpy as np
import jax.numpy as jnp
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from paz.models.foundation.dinov3.layers.ffn_layers import (
    Mlp,
    SwiGLUFFN,
)

from torch_layers_for_testing import (
    PT_Mlp,
    PT_SwiGLUFFN,
)


def create_mlp_pair(
    in_dim,
    drop=0.0,
    bias=True,
    hidden_features=None,
    out_features=None,
    approximate_gelu=False,
):
    """Creates, builds, and transfers weights for a Keras and PyTorch Mlp pair."""
    # 1. Create and configure PyTorch model
    pt_act_layer = lambda: nn.GELU(approximate="tanh" if approximate_gelu else "none")
    pt_model = PT_Mlp(
        in_features=in_dim,
        hidden_features=hidden_features,
        out_features=out_features,
        act_layer=pt_act_layer,
        drop=drop,
        bias=bias,
    )
    pt_model.eval()

    # 2. Create and build Keras model
    keras_model = Mlp(
        hidden_features=hidden_features,
        out_features=out_features,
        drop=drop,
        bias=bias,
        approximate_gelu=approximate_gelu,
    )
    keras_model(jnp.ones((1, 1, in_dim)))

    # ... rest of the weight transfer is the same
    pt_weights = pt_model.state_dict()
    fc1_weights = [pt_weights["fc1.weight"].T.numpy()]
    if bias:
        fc1_weights.append(pt_weights["fc1.bias"].numpy())
    keras_model.fc1.set_weights(fc1_weights)

    fc2_weights = [pt_weights["fc2.weight"].T.numpy()]
    if bias:
        fc2_weights.append(pt_weights["fc2.bias"].numpy())
    keras_model.fc2.set_weights(fc2_weights)

    return pt_model, keras_model


def create_swiglu_pair(
    in_dim, bias=True, align_to=8, hidden_features=None, out_features=None
):
    """Creates, builds, and transfers weights for a Keras and PyTorch SwiGLUFFN pair."""
    # 1. Create and configure PyTorch model
    pt_model = PT_SwiGLUFFN(
        in_features=in_dim,
        hidden_features=hidden_features,
        out_features=out_features,
        bias=bias,
        align_to=align_to,
    )
    pt_model.eval()  # Set to evaluation mode

    # 2. Create and build Keras model
    keras_model = SwiGLUFFN(
        hidden_features=hidden_features,
        out_features=out_features,
        bias=bias,
        align_to=align_to,
    )
    keras_model(jnp.ones((1, 1, in_dim)))  # Build the layer

    # 3. Transfer weights from PyTorch to Keras
    pt_weights = pt_model.state_dict()
    for i in range(1, 4):
        layer_name = f"w{i}"
        keras_layer = getattr(keras_model, layer_name)
        pt_layer_weights = [pt_weights[f"{layer_name}.weight"].T.numpy()]
        if bias:
            pt_layer_weights.append(pt_weights[f"{layer_name}.bias"].numpy())
        keras_layer.set_weights(pt_layer_weights)

    return pt_model, keras_model


@pytest.fixture(scope="module")
def params():
    """Provides a dictionary of common parameters for tests."""
    # Set a fixed seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    return {
        "batch_size": 4,
        "seq_len": 16,
        "in_dim": 64,
        "hidden_dim": 128,
        "drop_rate": 0.5,
    }


def test_mlp_basic(params):
    """🧪 Test Mlp: Basic case with specified dimensions."""
    pt_mlp, keras_mlp = create_mlp_pair(
        in_dim=params["in_dim"], hidden_features=params["hidden_dim"]
    )
    input_pt = torch.randn(params["batch_size"], params["seq_len"], params["in_dim"])
    output_pt = pt_mlp(input_pt).detach().numpy()
    output_keras = keras_mlp(jnp.array(input_pt.numpy()), training=False)
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_mlp_defaults(params):
    """🧪 Test Mlp: Defaulting hidden and out features to input dimension."""
    pt_mlp, keras_mlp = create_mlp_pair(in_dim=params["in_dim"])
    input_pt = torch.randn(params["batch_size"], params["seq_len"], params["in_dim"])
    output_pt = pt_mlp(input_pt).detach().numpy()
    output_keras = keras_mlp(jnp.array(input_pt.numpy()), training=False)
    assert output_keras.shape[-1] == params["in_dim"]
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_mlp_no_bias(params):
    """🧪 Test Mlp: Operation without bias terms."""
    pt_mlp, keras_mlp = create_mlp_pair(
        in_dim=params["in_dim"], hidden_features=params["hidden_dim"], bias=False
    )
    input_pt = torch.randn(params["batch_size"], params["seq_len"], params["in_dim"])
    output_pt = pt_mlp(input_pt).detach().numpy()
    output_keras = keras_mlp(jnp.array(input_pt.numpy()), training=False)
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_mlp_dropout_inference(params):
    """🧪 Test Mlp: Ensure dropout is handled correctly during inference."""
    # Create models with a non-zero dropout rate
    pt_mlp, keras_mlp = create_mlp_pair(
        in_dim=params["in_dim"],
        hidden_features=params["hidden_dim"],
        drop=params["drop_rate"],
    )

    # Generate a random input tensor
    input_pt = torch.randn(params["batch_size"], params["seq_len"], params["in_dim"])

    # Get the output from the PyTorch model (in eval mode, dropout is off)
    output_pt = pt_mlp(input_pt).detach().numpy()

    # Get the output from the Keras model with training=False to disable dropout
    output_keras = keras_mlp(jnp.array(input_pt.numpy()), training=False)

    # Assert that the outputs are nearly identical
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_swiglu_basic(params):
    """🧪 Test SwiGLUFFN: Basic case with specified dimensions."""
    pt_swiglu, keras_swiglu = create_swiglu_pair(
        in_dim=params["in_dim"], hidden_features=params["hidden_dim"]
    )
    input_pt = torch.randn(params["batch_size"], params["seq_len"], params["in_dim"])
    output_pt = pt_swiglu(input_pt).detach().numpy()
    output_keras = keras_swiglu(jnp.array(input_pt.numpy()))
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_swiglu_defaults(params):
    """🧪 Test SwiGLUFFN: Defaulting hidden and out features to input dimension."""
    pt_swiglu, keras_swiglu = create_swiglu_pair(in_dim=params["in_dim"])
    input_pt = torch.randn(params["batch_size"], params["seq_len"], params["in_dim"])
    output_pt = pt_swiglu(input_pt).detach().numpy()
    output_keras = keras_swiglu(jnp.array(input_pt.numpy()))
    assert output_keras.shape[-1] == params["in_dim"]
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_swiglu_no_bias(params):
    """🧪 Test SwiGLUFFN: Operation without bias terms."""
    pt_swiglu, keras_swiglu = create_swiglu_pair(
        in_dim=params["in_dim"], hidden_features=params["hidden_dim"], bias=False
    )
    input_pt = torch.randn(params["batch_size"], params["seq_len"], params["in_dim"])
    output_pt = pt_swiglu(input_pt).detach().numpy()
    output_keras = keras_swiglu(jnp.array(input_pt.numpy()))
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_swiglu_alignment(params):
    """🧪 Test SwiGLUFFN: Non-default alignment value."""
    pt_swiglu, keras_swiglu = create_swiglu_pair(
        in_dim=params["in_dim"], hidden_features=params["hidden_dim"], align_to=16
    )
    input_pt = torch.randn(params["batch_size"], params["seq_len"], params["in_dim"])
    output_pt = pt_swiglu(input_pt).detach().numpy()
    output_keras = keras_swiglu(jnp.array(input_pt.numpy()))
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)
