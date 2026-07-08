import numpy as np
import keras
from keras import Input, Model

from paz.models.foundation.dinov2.layers.block import block, apply_feedforward


def make_block_args(name="block_0"):
    return dict(
        dim=24,
        num_heads=4,
        use_qkv_bias=True,
        use_projection_bias=True,
        attention_dropout_rate=0.0,
        feedforward_layer="mlp",
        mlp_ratio=2,
        use_feedforward_bias=True,
        activation="gelu",
        drop_rate=0.0,
        init_values=None,
        drop_path=0.0,
        normalization_layer=build_layer_norm,
        name=name,
    )


def build_layer_norm(name):
    return keras.layers.LayerNormalization(epsilon=1e-6, name=name)


def build_block_functional_model(name="block_0"):
    args = make_block_args(name)
    inputs = Input(shape=(None, args["dim"]), name=f"{name}_input")
    outputs = block(inputs, **args)
    return Model(inputs, outputs, name=name)


def test_block_returns_tensor_with_input_shape():
    model = build_block_functional_model()
    x = np.random.randn(2, 9, 24).astype(np.float32)
    y = np.array(model(x))
    assert y.shape == (2, 9, 24)


def test_block_has_expected_sublayer_names():
    model = build_block_functional_model()
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


def test_apply_feedforward_mlp_branch_returns_tensor():
    inputs = Input(shape=(None, 24))
    args = (inputs, 24, "mlp", 2, True, "gelu", 0.0, "ffn_branch")
    output = apply_feedforward(*args)
    assert output.shape[-1] == 24


def test_apply_feedforward_swiglu_branch_returns_tensor():
    inputs = Input(shape=(None, 24))
    args = (inputs, 24, "swiglu", 2, True, "silu", 0.0, "swiglu_branch")
    output = apply_feedforward(*args)
    assert output.shape[-1] == 24
