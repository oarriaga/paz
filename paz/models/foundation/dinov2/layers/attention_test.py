import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import keras
from keras import Input, Model

from paz.models.foundation.dinov2.layers.attention import attend
from paz.models.foundation.dinov2.layers.block import (
    block,
    AttentionArgs,
    FFNArgs,
)


def build_attention_model(
    seq_len, dim, num_heads, head_dim, attn_drop, proj_drop, qkv_bias, proj_bias
):
    inputs = Input(shape=(seq_len, dim), name="attn_input")
    args = (
        inputs,
        num_heads,
        head_dim,
        attn_drop,
        proj_drop,
        qkv_bias,
        proj_bias,
        "attn",
    )
    outputs = attend(*args)
    return Model(inputs=inputs, outputs=outputs, name="attention_model")


def test_attend_output_shape():
    model = build_attention_model(7, 12, 3, 4, 0.0, 0.0, True, True)
    x = np.random.randn(2, 7, 12).astype(np.float32)
    y = model(x)
    assert y.shape == (2, 7, 12)


def test_attend_no_nan_with_dropout_zero():
    model = build_attention_model(5, 16, 4, 4, 0.0, 0.0, True, True)
    x = np.random.randn(1, 5, 16).astype(np.float32)
    y = np.array(model(x))
    assert not np.any(np.isnan(y))


def test_build_block_output_shape():
    args = build_block_args()
    model = build_block_model(args)
    x = np.random.randn(2, 9, 24).astype(np.float32)
    y = model(x)
    assert y.shape == (2, 9, 24)


def test_build_block_layer_names_present():
    args = build_block_args()
    model = build_block_model(args)
    expected = [
        "block_0_norm1",
        "block_0_qkv",
        "block_0_proj",
        "block_0_norm2",
        "block_0_mlp_fc1",
        "block_0_mlp_fc2",
        "block_0_add1",
        "block_0_add2",
    ]
    names = [layer.name for layer in model.layers]
    for layer_name in expected:
        assert layer_name in names


def build_block_model(args):
    inputs = Input(shape=(None, args["dim"]), name=f"{args['name']}_input")
    outputs = block(inputs, **args)
    return Model(inputs, outputs, name=args["name"])


def test_attend_parity_with_pytorch_dinov2():
    pytest = __import__("pytest")
    torch = __import__("torch")
    try:
        ref = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
        )
    except Exception:
        pytest.skip("torch hub unavailable")
    ref.eval()
    state = ref.state_dict()
    qkv_w = state["blocks.0.attn.qkv.weight"].numpy()
    qkv_b = state["blocks.0.attn.qkv.bias"].numpy()
    proj_w = state["blocks.0.attn.proj.weight"].numpy()
    proj_b = state["blocks.0.attn.proj.bias"].numpy()
    np.random.seed(0)
    torch.manual_seed(0)
    x_np = np.random.randn(1, 5, 384).astype(np.float32)
    model = build_attention_model(5, 384, 6, 64, 0.0, 0.0, True, True)
    model(x_np)
    model.get_layer("attn_qkv").set_weights([qkv_w.T, qkv_b])
    model.get_layer("attn_proj").set_weights([proj_w.T, proj_b])
    keras_out = np.array(model(x_np))
    ref_attn = ref.blocks[0].attn
    with torch.no_grad():
        ref_out = ref_attn(torch.from_numpy(x_np)).numpy()
    np.testing.assert_allclose(keras_out, ref_out, atol=1e-4, rtol=1e-4)


def build_block_args():
    attention = AttentionArgs(4, True, True, 0.0)
    FFN = FFNArgs("mlp", 2, True, "gelu")
    return dict(
        dim=24,
        attention=attention,
        FFN=FFN,
        drop_rate=0.0,
        init_values=None,
        drop_path=0.0,
        normalization_layer=lambda name: keras.layers.LayerNormalization(
            epsilon=1e-6, name=name
        ),
        name="block_0",
    )
