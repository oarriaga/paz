import os
import pytest
import torch
import numpy as np
from keras.layers import LayerNormalization
from torch import nn

from paz.models.foundation.dinov3.layers.patch_embed import PatchEmbed
from paz.models.foundation.dinov3.layers.torch_layers_for_testing import PT_PatchEmbed


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


DINO_REPO_PATH = ""  # Not needed
DINO_WEIGHT_PATH = "/path/that/does/not/exist/dinov3_vits16_pretrain.pth"
dinov3_files_exist = os.path.isfile(DINO_WEIGHT_PATH)


@pytest.mark.skipif(
    not dinov3_files_exist, reason="DINOv3 local model/weight files not found."
)
def test_dinov3_pretrained_weights_match(params):
    """Tests the layer by loading weights from a pre-trained DINOv3 model."""
    # 1. Load the pre-trained DINOv3 model
    dinov3_model = torch.hub.load(
        DINO_REPO_PATH, "dinov3_vits16", source="local", weights=DINO_WEIGHT_PATH
    )
    dinov3_model.eval()
    torch_pretrained_pe = dinov3_model.patch_embed

    # 2. Configure the Keras layer to match the pre-trained model
    conv_has_bias = torch_pretrained_pe.proj.bias is not None
    keras_norm_layer = None
    if isinstance(torch_pretrained_pe.norm, nn.LayerNorm):
        epsilon = torch_pretrained_pe.norm.eps
        keras_norm_layer = lambda **kwargs: LayerNormalization(
            epsilon=epsilon, **kwargs
        )

    keras_pe_pretrained = PatchEmbed(
        img_size=torch_pretrained_pe.img_size,
        patch_size=torch_pretrained_pe.patch_size,
        in_channels=torch_pretrained_pe.in_chans,
        embed_dim=torch_pretrained_pe.embed_dim,
        norm_layer=keras_norm_layer,
        flatten_embedding=torch_pretrained_pe.flatten_embedding,
        use_bias=conv_has_bias,
    )

    # 3. Create identical random input data
    img_h, img_w = torch_pretrained_pe.img_size
    x_keras_pretrained = np.random.rand(
        params["BATCH_SIZE"], img_h, img_w, torch_pretrained_pe.in_chans
    ).astype("float32")
    x_torch_pretrained = torch.from_numpy(
        np.transpose(x_keras_pretrained, (0, 3, 1, 2))
    )

    # 4. Build Keras model, port weights, and compare outputs
    _ = keras_pe_pretrained(x_keras_pretrained)
    port_weights(torch_pretrained_pe, keras_pe_pretrained)

    torch_out = torch_pretrained_pe(x_torch_pretrained).detach().cpu().numpy()
    keras_out = np.array(keras_pe_pretrained(x_keras_pretrained))

    np.testing.assert_allclose(
        torch_out,
        keras_out,
        atol=1e-5,
        err_msg="Final outputs with DINOv3 weights do not match.",
    )
