import pytest

torch = pytest.importorskip("torch")
from torch import nn

import numpy as np
from keras.layers import LayerNormalization

from paz.models.foundation.dinov3.layers.patch_embed import PatchEmbed
from torch_layers_for_testing import PT_PatchEmbed


def port_weights(torch_model, keras_model):
    """Port weights from PyTorch PatchEmbed to Keras PatchEmbed."""
    # Port convolution weights (OIHW -> HWIO)
    torch_kernel = torch_model.proj.weight.detach().cpu().numpy()
    keras_model.projection.kernel.assign(np.transpose(torch_kernel, (2, 3, 1, 0)))

    # Port convolution bias if it exists
    if torch_model.proj.bias is not None:
        keras_model.projection.bias.assign(torch_model.proj.bias.detach().cpu().numpy())

    # Port LayerNorm weights if it's not an Identity layer
    if isinstance(torch_model.norm, nn.LayerNorm):
        keras_model.norm.gamma.assign(torch_model.norm.weight.detach().cpu().numpy())
        keras_model.norm.beta.assign(torch_model.norm.bias.detach().cpu().numpy())


@pytest.fixture(scope="module")
def params():
    """Provides common parameters and sets random seeds."""
    torch.manual_seed(0)
    np.random.seed(0)
    return {
        "IMG_SIZE": 224,
        "PATCH_SIZE": 16,
        "IN_CHANNELS": 3,
        "EMBED_DIM": 384,
        "BATCH_SIZE": 4,
    }


@pytest.fixture
def input_data(params):
    """Generates matching numpy (Keras) and torch input tensors."""
    p = params
    # Keras input: (B, H, W, C)
    keras_input = np.random.rand(
        p["BATCH_SIZE"], p["IMG_SIZE"], p["IMG_SIZE"], p["IN_CHANNELS"]
    ).astype("float32")
    # PyTorch input: (B, C, H, W)
    torch_input = torch.from_numpy(np.transpose(keras_input, (0, 3, 1, 2)))
    return keras_input, torch_input


def test_basic_functionality(params, input_data):
    """Tests the PatchEmbed layer without Layer Normalization."""
    p = params
    keras_in, torch_in = input_data

    # Initialize models
    torch_pe = PT_PatchEmbed(
        p["IMG_SIZE"], p["PATCH_SIZE"], p["IN_CHANNELS"], p["EMBED_DIM"]
    )
    keras_pe = PatchEmbed(
        p["IMG_SIZE"], p["PATCH_SIZE"], p["IN_CHANNELS"], p["EMBED_DIM"]
    )

    # Build Keras layer and port weights
    _ = keras_pe(keras_in)
    port_weights(torch_pe, keras_pe)

    # Compare outputs
    torch_out = torch_pe(torch_in).detach().cpu().numpy()
    keras_out = np.array(keras_pe(keras_in))

    np.testing.assert_allclose(torch_out, keras_out, atol=1e-5)


def test_with_layer_normalization(params, input_data):
    """Tests the PatchEmbed layer with Layer Normalization."""
    p = params
    keras_in, torch_in = input_data

    # Initialize models with LayerNorm
    torch_pe = PT_PatchEmbed(
        p["IMG_SIZE"],
        p["PATCH_SIZE"],
        p["IN_CHANNELS"],
        p["EMBED_DIM"],
        norm_layer=nn.LayerNorm,
    )
    keras_pe = PatchEmbed(
        p["IMG_SIZE"],
        p["PATCH_SIZE"],
        p["IN_CHANNELS"],
        p["EMBED_DIM"],
        norm_layer=lambda **kwargs: LayerNormalization(epsilon=1e-5, **kwargs),
    )

    _ = keras_pe(keras_in)
    port_weights(torch_pe, keras_pe)

    torch_out = torch_pe(torch_in).detach().cpu().numpy()
    keras_out = np.array(keras_pe(keras_in))

    np.testing.assert_allclose(torch_out, keras_out, atol=1e-5)


def test_flatten_embedding_false(params, input_data):
    """Tests the layer's output shape when flatten_embedding is False."""
    p = params
    keras_in, torch_in = input_data

    torch_pe = PT_PatchEmbed(
        p["IMG_SIZE"],
        p["PATCH_SIZE"],
        p["IN_CHANNELS"],
        p["EMBED_DIM"],
        flatten_embedding=False,
    )
    keras_pe = PatchEmbed(
        p["IMG_SIZE"],
        p["PATCH_SIZE"],
        p["IN_CHANNELS"],
        p["EMBED_DIM"],
        flatten_embedding=False,
    )

    _ = keras_pe(keras_in)
    port_weights(torch_pe, keras_pe)

    torch_out = torch_pe(torch_in).detach().cpu().numpy()
    keras_out = np.array(keras_pe(keras_in))

    # Check shape: (B, H_new, W_new, C_new)
    expected_h = p["IMG_SIZE"] // p["PATCH_SIZE"]
    expected_w = p["IMG_SIZE"] // p["PATCH_SIZE"]
    assert keras_out.shape == (p["BATCH_SIZE"], expected_h, expected_w, p["EMBED_DIM"])
    np.testing.assert_allclose(torch_out, keras_out, atol=1e-5)
