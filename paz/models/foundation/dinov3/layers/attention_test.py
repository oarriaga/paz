import os
import sys
import pytest
import torch
import numpy as np

os.environ["KERAS_BACKEND"] = "jax"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==============================================================================
# Keras Layer Implementation
# ==============================================================================
from paz.models.foundation.dinov3.layers.attention import (
    SelfAttention,
    CausalSelfAttention,
)


# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

from paz.models.foundation.dinov3.layers.torch_layers_for_testing import (
    PT_SelfAttention,
    PT_CausalSelfAttention,
)

# ==============================================================================
# Utility Functions
# ==============================================================================


def port_weights(torch_model, keras_model):
    """Copies weights from a PyTorch model to its Keras equivalent."""
    keras_model.qkv.kernel.assign(torch_model.qkv.weight.T.detach().numpy())
    if torch_model.qkv.bias is not None:
        keras_model.qkv.bias.assign(torch_model.qkv.bias.detach().numpy())

    keras_model.proj.kernel.assign(torch_model.proj.weight.T.detach().numpy())
    if torch_model.proj.bias is not None:
        keras_model.proj.bias.assign(torch_model.proj.bias.detach().numpy())


# ==============================================================================
# PyTest Fixtures and Test Cases
# ==============================================================================


@pytest.fixture(scope="module")
def params():
    """Provides common parameters for the tests."""
    torch.manual_seed(0)
    np.random.seed(0)
    return {"DIM": 128, "NUM_HEADS": 4, "BATCH": 2, "SEQ_LEN": 64}


@pytest.fixture(scope="module")
def inputs(params):
    """Provides random input tensors for Torch and Keras."""
    p = params
    x_torch = torch.rand(p["BATCH"], p["SEQ_LEN"], p["DIM"])
    rope_torch = (
        torch.rand(1, p["NUM_HEADS"], p["SEQ_LEN"], p["DIM"] // p["NUM_HEADS"]),
        torch.rand(1, p["NUM_HEADS"], p["SEQ_LEN"], p["DIM"] // p["NUM_HEADS"]),
    )
    x_keras = x_torch.numpy()
    rope_keras = (rope_torch[0].numpy(), rope_torch[1].numpy())
    return x_torch, x_keras, rope_torch, rope_keras


@pytest.fixture(scope="module")
def self_attention_models(params):
    """Provides initialized and weight-ported SelfAttention models."""
    p = params
    torch_attn = PT_SelfAttention(
        p["DIM"], p["NUM_HEADS"], qkv_bias=True, proj_bias=True
    )
    keras_attn = SelfAttention(p["DIM"], p["NUM_HEADS"], qkv_bias=True, proj_bias=True)
    torch_attn.eval()

    # Build layer and port weights
    _ = keras_attn(np.zeros((p["BATCH"], p["SEQ_LEN"], p["DIM"]), dtype="float32"))
    port_weights(torch_attn, keras_attn)
    return torch_attn, keras_attn


@pytest.fixture(scope="module")
def causal_self_attention_models(params):
    """Provides initialized and weight-ported CausalSelfAttention models."""
    p = params
    torch_causal_attn = PT_CausalSelfAttention(
        p["DIM"], p["NUM_HEADS"], qkv_bias=True, proj_bias=True
    )
    keras_causal_attn = CausalSelfAttention(
        p["DIM"], p["NUM_HEADS"], qkv_bias=True, proj_bias=True
    )
    torch_causal_attn.eval()

    # Build layer and port weights
    _ = keras_causal_attn(
        np.zeros((p["BATCH"], p["SEQ_LEN"], p["DIM"]), dtype="float32")
    )
    port_weights(torch_causal_attn, keras_causal_attn)
    return torch_causal_attn, keras_causal_attn


# --- Test Cases ---
def test_self_attention_call(self_attention_models, inputs):
    """Tests the `call` method of SelfAttention with RoPE."""
    torch_attn, keras_attn = self_attention_models
    x_torch, x_keras, rope_torch, rope_keras = inputs

    torch_out = torch_attn(x_torch, rope=rope_torch).detach().numpy()
    keras_out = np.array(keras_attn(x_keras, rope=rope_keras, training=False))

    np.testing.assert_allclose(
        torch_out, keras_out, atol=1e-5, err_msg="`call` method outputs differ."
    )


def test_self_attention_forward_list(self_attention_models, inputs):
    """Tests the `forward_list` method of SelfAttention."""
    torch_attn, keras_attn = self_attention_models
    x_torch, x_keras, rope_torch, rope_keras = inputs

    torch_out_list = torch_attn.forward_list(
        x_list=[x_torch, x_torch], rope_list=[rope_torch, rope_torch]
    )
    keras_out_list = keras_attn.forward_list(
        [x_keras, x_keras], rope_list=[rope_keras, rope_keras], training=False
    )

    np.testing.assert_allclose(
        torch_out_list[0].detach().numpy(),
        np.array(keras_out_list[0]),
        atol=1e-5,
        err_msg="`forward_list` method outputs differ.",
    )


def test_causal_self_attention_call(causal_self_attention_models, inputs):
    """Tests the `call` method of CausalSelfAttention."""
    torch_causal_attn, keras_causal_attn = causal_self_attention_models
    x_torch, x_keras, _, _ = inputs

    torch_out = torch_causal_attn(x_torch).detach().numpy()
    keras_out = np.array(keras_causal_attn(x_keras, training=False))

    np.testing.assert_allclose(
        torch_out, keras_out, atol=1e-5, err_msg="Causal `call` method outputs differ."
    )


def test_self_attention_without_rope(self_attention_models, inputs):
    """Tests the SelfAttention layer without applying RoPE."""
    torch_attn, keras_attn = self_attention_models
    x_torch, x_keras, _, _ = inputs

    torch_out = torch_attn(x_torch, rope=None).detach().numpy()
    keras_out = np.array(keras_attn(x_keras, rope=None, training=False))

    np.testing.assert_allclose(
        torch_out, keras_out, atol=1e-5, err_msg="`call` method without RoPE failed."
    )


def test_causal_attention_as_standard_attention(causal_self_attention_models, inputs):
    """Tests CausalSelfAttention with `is_causal=False` to ensure it matches standard attention."""
    torch_causal_attn, keras_causal_attn = causal_self_attention_models
    x_torch, x_keras, _, _ = inputs

    torch_out = torch_causal_attn(x_torch, is_causal=False).detach().numpy()
    keras_out = np.array(keras_causal_attn(x_keras, is_causal=False, training=False))

    np.testing.assert_allclose(
        torch_out,
        keras_out,
        atol=1e-5,
        err_msg="Causal layer with `is_causal=False` failed.",
    )


def test_dropout_activation(params, inputs):
    """Tests that dropout layers are active during training and inactive during evaluation."""
    p = params
    _, x_keras, _, _ = inputs
    keras_attn_dropout = SelfAttention(
        p["DIM"], p["NUM_HEADS"], attn_drop=0.5, proj_drop=0.5
    )
    _ = keras_attn_dropout(x_keras)

    out_train = keras_attn_dropout(x_keras, training=True)
    out_eval = keras_attn_dropout(x_keras, training=False)

    assert not np.allclose(
        out_train, out_eval
    ), "Dropout layers seem inactive during training."


def test_forward_list_varied_shapes(self_attention_models, params, inputs):
    """Tests `forward_list` with a list of tensors having different sequence lengths."""
    torch_attn, keras_attn = self_attention_models
    p = params
    x_torch, x_keras, rope_torch, rope_keras = inputs

    x_torch_short = torch.rand(p["BATCH"], p["SEQ_LEN"] // 2, p["DIM"])
    x_keras_short = x_torch_short.numpy()
    rope_torch_short = (
        torch.rand(1, p["NUM_HEADS"], p["SEQ_LEN"] // 2, p["DIM"] // p["NUM_HEADS"]),
        torch.rand(1, p["NUM_HEADS"], p["SEQ_LEN"] // 2, p["DIM"] // p["NUM_HEADS"]),
    )
    rope_keras_short = (rope_torch_short[0].numpy(), rope_torch_short[1].numpy())

    torch_list_varied = torch_attn.forward_list(
        x_list=[x_torch, x_torch_short], rope_list=[rope_torch, rope_torch_short]
    )
    keras_list_varied = keras_attn.forward_list(
        [x_keras, x_keras_short],
        rope_list=[rope_keras, rope_keras_short],
        training=False,
    )

    np.testing.assert_allclose(
        torch_list_varied[0].detach().numpy(), keras_list_varied[0], atol=1e-5
    )
    np.testing.assert_allclose(
        torch_list_varied[1].detach().numpy(), keras_list_varied[1], atol=1e-5
    )


def test_initialization_no_biases(params, inputs):
    """Tests model initialization and execution with biases set to False."""
    p = params
    x_torch, x_keras, rope_torch, rope_keras = inputs
    torch_attn_no_bias = PT_SelfAttention(
        p["DIM"], p["NUM_HEADS"], qkv_bias=False, proj_bias=False
    )
    keras_attn_no_bias = SelfAttention(
        p["DIM"], p["NUM_HEADS"], qkv_bias=False, proj_bias=False
    )
    torch_attn_no_bias.eval()

    _ = keras_attn_no_bias(x_keras)
    port_weights(torch_attn_no_bias, keras_attn_no_bias)

    torch_out = torch_attn_no_bias(x_torch, rope=rope_torch).detach().numpy()
    keras_out = np.array(keras_attn_no_bias(x_keras, rope=rope_keras, training=False))

    np.testing.assert_allclose(
        torch_out, keras_out, atol=1e-5, err_msg="Model with no biases failed."
    )


def test_value_error_on_invalid_heads(params):
    """Tests that a ValueError is raised if `dim` is not divisible by `num_heads`."""
    with pytest.raises(ValueError, match="must be divisible by"):
        SelfAttention(dim=params["DIM"], num_heads=7)


# --- DINOv3 Pre-trained Weights Test ---

DINO_REPO_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3"
DINO_WEIGHT_PATH = r"D:\DFKI_SeaMe_project\Tasks\Task2_porting_paz_model_to_keras3\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
dinov3_files_exist = os.path.isdir(DINO_REPO_PATH) and os.path.isfile(DINO_WEIGHT_PATH)


@pytest.mark.skipif(
    not dinov3_files_exist, reason="DINOv3 local model/weight files not found."
)
def test_dinov3_pretrained_weights_match():
    """
    Tests the Keras SelfAttention layer by loading weights from a pre-trained
    DINOv3 PyTorch model and verifying the outputs match.
    """
    # 1. Load the pre-trained DINOv3 model from local source
    dinov3_model_local = torch.hub.load(
        DINO_REPO_PATH, "dinov3_vits16", source="local", weights=DINO_WEIGHT_PATH
    )
    dinov3_model_local.eval()

    # 2. Extract the first attention layer and its parameters
    torch_pretrained_attn = dinov3_model_local.blocks[0].attn
    PRETRAINED_DIM = dinov3_model_local.embed_dim
    PRETRAINED_NUM_HEADS = torch_pretrained_attn.num_heads
    QKV_BIAS = torch_pretrained_attn.qkv.bias is not None
    PROJ_BIAS = torch_pretrained_attn.proj.bias is not None

    # 3. Initialize Keras SelfAttention with matching parameters
    keras_attn_pretrained = SelfAttention(
        dim=PRETRAINED_DIM,
        num_heads=PRETRAINED_NUM_HEADS,
        qkv_bias=QKV_BIAS,
        proj_bias=PROJ_BIAS,
    )

    # 4. Prepare a dummy input tensor
    BATCH_SIZE = 2
    NUM_TOKENS = 201
    x_torch_pretrained = torch.rand(BATCH_SIZE, NUM_TOKENS, PRETRAINED_DIM)
    x_keras_pretrained = x_torch_pretrained.numpy()

    # 5. Build Keras layer and port weights
    _ = keras_attn_pretrained(np.zeros_like(x_keras_pretrained, dtype="float32"))
    port_weights(torch_pretrained_attn, keras_attn_pretrained)

    # 6. Run forward pass on both layers and compare outputs
    torch_out_pretrained = torch_pretrained_attn(x_torch_pretrained).detach().numpy()
    keras_out_pretrained = np.array(
        keras_attn_pretrained(x_keras_pretrained, rope=None, training=False)
    )

    np.testing.assert_allclose(
        torch_out_pretrained,
        keras_out_pretrained,
        atol=1e-5,
        err_msg="Keras output with DINOv3 weights does not match PyTorch output.",
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
