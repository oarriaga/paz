import pytest

torch = pytest.importorskip("torch")

import numpy as np
from paz.models.foundation.dinov3.layers.attention import (
    SelfAttention,
    CausalSelfAttention,
)
from torch_layers_for_testing import (
    PT_SelfAttention,
    PT_CausalSelfAttention,
)


def port_weights(torch_model, keras_model):
    """Copies weights from a PyTorch model to its Keras equivalent."""
    keras_model.qkv.kernel.assign(torch_model.qkv.weight.T.detach().numpy())
    if torch_model.qkv.bias is not None:
        keras_model.qkv.bias.assign(torch_model.qkv.bias.detach().numpy())

    keras_model.proj.kernel.assign(torch_model.proj.weight.T.detach().numpy())
    if torch_model.proj.bias is not None:
        keras_model.proj.bias.assign(torch_model.proj.bias.detach().numpy())


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

    _ = keras_causal_attn(
        np.zeros((p["BATCH"], p["SEQ_LEN"], p["DIM"]), dtype="float32")
    )
    port_weights(torch_causal_attn, keras_causal_attn)
    return torch_causal_attn, keras_causal_attn


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
