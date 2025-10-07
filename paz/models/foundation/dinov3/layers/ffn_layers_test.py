import os
import torch
import numpy as np
import sys

os.environ["KERAS_BACKEND"] = "jax"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

import torch
import torch.nn as nn
import jax.numpy as jnp

import pytest


# ==============================================================================
# Keras Layer Implementation
# ==============================================================================

from paz.models.foundation.dinov3.layers.ffn_layers import (
    Mlp,
    SwiGLUFFN,
)

# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

from paz.models.foundation.dinov3.layers.torch_layers_for_testing import (
    PT_Mlp,
    PT_SwiGLUFFN,
)

# ==============================================================================
# Helper Functions for Creating and Testing Layer Pairs
# ==============================================================================


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


# ==============================================================================
# Test Fixtures and Cases
# ==============================================================================


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


# --- Test Cases for Mlp ---
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


def test_mlp_with_pretrained_dinov3_weights(params):
    """🧪 Test Mlp by loading real DINOv3 ViT-S/16 pretrained weights."""
    try:
        DINO_REPO_PATH = (
            r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
        )
        DINO_WEIGHT_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

        if not os.path.isdir(DINO_REPO_PATH) or not os.path.isfile(DINO_WEIGHT_PATH):
            raise FileNotFoundError("DINOv3 repository or weight file not found.")

        # Load the full DINOv3 model from the local repository
        full_dinov3_model = torch.hub.load(
            DINO_REPO_PATH, "dinov3_vits16", source="local", pretrained=False
        )
        state_dict = torch.load(DINO_WEIGHT_PATH, map_location=torch.device("cpu"))
        full_dinov3_model.load_state_dict(state_dict)
        full_dinov3_model.eval()

        # Extract the first MLP block as the reference PyTorch model
        torch_model = full_dinov3_model.blocks[0].mlp

    except (FileNotFoundError, ModuleNotFoundError, ImportError, AttributeError) as e:
        pytest.skip(
            f"DINOv3 assets not found, skipping pretrained MLP test. Reason: {e}"
        )

    # Get model dimensions from the loaded torch model
    dinov3_embed_dim = torch_model.fc1.in_features
    dinov3_hidden_dim = torch_model.fc1.out_features

    # Create the Keras model with a matching configuration
    keras_model = Mlp(
        hidden_features=dinov3_hidden_dim,
        out_features=dinov3_embed_dim,
        approximate_gelu=False,  # Use precise GELU to match PyTorch
    )
    # Build the Keras layer by calling it with a dummy input
    keras_model(jnp.ones((1, 1, dinov3_embed_dim)))

    # Transfer weights from the PyTorch layer to the Keras layer
    keras_model.fc1.set_weights(
        [
            torch_model.fc1.weight.T.detach().numpy(),
            torch_model.fc1.bias.detach().numpy(),
        ]
    )
    keras_model.fc2.set_weights(
        [
            torch_model.fc2.weight.T.detach().numpy(),
            torch_model.fc2.bias.detach().numpy(),
        ]
    )

    # --- Run forward pass and compare outputs ---
    input_tensor = torch.randn(
        params["batch_size"], params["seq_len"], dinov3_embed_dim
    )
    output_pt = torch_model(input_tensor).detach().numpy()
    output_keras = keras_model(jnp.array(input_tensor.numpy()), training=False)

    # Calculate and validate the mean absolute difference
    mean_abs_diff = np.mean(np.abs(output_pt - output_keras))
    print(f"\nMean Absolute Difference: {mean_abs_diff}")

    # Assert that the outputs are numerically very close
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-4)
    assert (
        mean_abs_diff < 1e-5
    ), f"Mean absolute difference {mean_abs_diff} exceeds tolerance."


# --- Test Cases for swiglu ---
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


def test_swiglu_with_pretrained_dinov3_weights(params):
    """🧪 Test SwiGLUFFN with mock DINOv3 ViT-S/16 pretrained weights."""
    dinov3_embed_dim = 384
    dinov3_hidden_dim = 1536  # DINOv3 ViT-S uses an FFN size of 1536

    # 1. Create mock pre-trained weights using the PyTorch layer's structure
    torch_model_for_weights = PT_SwiGLUFFN(
        in_features=dinov3_embed_dim, hidden_features=dinov3_hidden_dim
    )
    mock_weights = torch_model_for_weights.state_dict()

    # 2. Configure the actual PyTorch model and load weights
    torch_model = PT_SwiGLUFFN(
        in_features=dinov3_embed_dim, hidden_features=dinov3_hidden_dim
    )
    torch_model.load_state_dict(mock_weights)
    torch_model.eval()

    keras_model = SwiGLUFFN(
        hidden_features=dinov3_hidden_dim, out_features=dinov3_embed_dim
    )
    keras_model(jnp.ones((1, 1, dinov3_embed_dim)))
    keras_model.w1.set_weights(
        [mock_weights["w1.weight"].T.numpy(), mock_weights["w1.bias"].numpy()]
    )
    keras_model.w2.set_weights(
        [mock_weights["w2.weight"].T.numpy(), mock_weights["w2.bias"].numpy()]
    )
    keras_model.w3.set_weights(
        [mock_weights["w3.weight"].T.numpy(), mock_weights["w3.bias"].numpy()]
    )

    input_tensor = torch.randn(
        params["batch_size"], params["seq_len"], dinov3_embed_dim
    )
    output_pt = torch_model(input_tensor).detach().numpy()
    output_keras = keras_model(jnp.array(input_tensor.numpy()))
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
