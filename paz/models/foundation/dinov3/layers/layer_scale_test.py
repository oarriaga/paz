import os
import sys
import pytest
import torch
import numpy as np

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==============================================================================
# Keras Layer Implementation
# ==============================================================================

from paz.models.foundation.dinov3.layers.layer_scale import LayerScale

# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

from paz.models.foundation.dinov3.layers.torch_layers_for_testing import PT_LayerScale


# ==============================================================================
# Helper Functions and Fixtures
# ==============================================================================
def port_weights(torch_model, keras_model):
    """Copies the gamma weight from the PyTorch model to the Keras model."""
    keras_model.gamma.assign(torch_model.gamma.detach().cpu().numpy())


# ==============================================================================
# Test Suite for LayerScale
# ==============================================================================


# --- Pytest Fixtures ---
@pytest.fixture(scope="module")
def params():
    """Provides common parameters for the tests and sets random seeds."""
    torch.manual_seed(0)
    np.random.seed(0)
    return {"DIM": 128, "BATCH": 4, "SEQ_LEN": 64}


# --- Test Cases ---
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


# --- DINOv3 Pre-trained Weights Test ---
DINO_REPO_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
DINO_WEIGHT_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
dinov3_files_exist = os.path.isdir(DINO_REPO_PATH) and os.path.isfile(DINO_WEIGHT_PATH)


@pytest.mark.skipif(
    not dinov3_files_exist, reason="DINOv3 local model/weight files not found."
)
def test_dinov3_pretrained_weights_match(params):
    """
    Tests LayerScale by loading weights from a pre-trained DINOv3 model
    and verifying the outputs match the original PyTorch layer.
    """
    # 1. Load the full pre-trained DINOv3 model
    dinov3_model = torch.hub.load(
        DINO_REPO_PATH, "dinov3_vits16", source="local", weights=DINO_WEIGHT_PATH
    )
    dinov3_model.eval()

    # 2. Extract a real LayerScale module from the first block
    torch_pretrained_ls = dinov3_model.blocks[0].ls1
    PRETRAINED_DIM = torch_pretrained_ls.gamma.shape[0]

    # 3. Initialize the Keras layer
    keras_ls_pretrained = LayerScale(dimension=PRETRAINED_DIM)

    # 4. Prepare a matching random input
    x_torch_pretrained = torch.rand(params["BATCH"], params["SEQ_LEN"], PRETRAINED_DIM)
    x_keras_pretrained = x_torch_pretrained.cpu().numpy()

    # 5. Build Keras layer and port the pre-trained weights
    _ = keras_ls_pretrained(x_keras_pretrained)
    port_weights(torch_pretrained_ls, keras_ls_pretrained)

    # 6. Get outputs and compare
    torch_out_pretrained = (
        torch_pretrained_ls(x_torch_pretrained).detach().cpu().numpy()
    )
    keras_out_pretrained = keras_ls_pretrained(x_keras_pretrained)

    np.testing.assert_allclose(
        torch_out_pretrained,
        keras_out_pretrained,
        atol=1e-6,
        err_msg="Pre-trained DINOv3 weights test failed.",
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
