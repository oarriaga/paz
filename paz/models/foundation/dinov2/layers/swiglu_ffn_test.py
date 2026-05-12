import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest
import keras
from keras import ops

from paz.models.foundation.dinov2.layers.swiglu_ffn import (
    SwiGLUFFN,
    SwiGLUFFNFused,
    SwiGLUFFNAligned,
    compute_swiglu_hidden,
    compute_effective_dims_standard,
    compute_effective_dims_fused,
    compute_effective_dims_aligned,
    build_dense,
    build_swiglu_ffn_layers,
    build_swiglu_aligned_layers,
)

# ─── Helpers ────────────────────────────────────────────────────────────────


def make_x(batch=2, seq=8, dim=64):
    arr = np.random.randn(batch, seq, dim).astype("float32")
    return ops.convert_to_tensor(arr)


def make_fused(dim=64, hid=256, **kw):
    return SwiGLUFFNFused(input_features=dim, hidden_features=hid, **kw)


def make_standard(dim=64, hid=256, **kw):
    return SwiGLUFFN(input_features=dim, hidden_features=hid, **kw)


def make_aligned(dim=64, hid=256, align_to=8, **kw):
    return SwiGLUFFNAligned(
        input_features=dim, hidden_features=hid, align_to=align_to, **kw
    )


# ─── compute_swiglu_hidden ───────────────────────────────────────────────────


class TestComputeSwiGLUHidden:
    def test_output_shape(self):
        act = keras.layers.Activation("silu")
        x = ops.ones((2, 8, 32))
        out = compute_swiglu_hidden(act, x)
        assert out.shape == (2, 8, 16)

    def test_splits_in_half(self):
        act = lambda v: v
        x = ops.ones((1, 4, 8))
        out = compute_swiglu_hidden(act, x)
        assert out.shape[-1] == 4

    def test_gate_applied(self):
        act = lambda v: v
        val_part = np.ones((1, 2, 4), dtype="float32")
        gate_part = np.zeros((1, 2, 4), dtype="float32")
        x = ops.convert_to_tensor(np.concatenate([val_part, gate_part], axis=-1))
        out = compute_swiglu_hidden(act, x)
        np.testing.assert_allclose(np.array(out), np.zeros((1, 2, 4)), atol=1e-6)

    def test_silu_activation(self):
        act = keras.layers.Activation("silu")
        x = ops.ones((1, 2, 4))
        out = compute_swiglu_hidden(act, x)
        assert out.shape == (1, 2, 2)

    def test_values_are_activation_times_gate(self):
        act = lambda v: v * 2
        v = np.ones((1, 1, 2), dtype="float32") * 3.0
        g = np.ones((1, 1, 2), dtype="float32") * 5.0
        x = ops.convert_to_tensor(np.concatenate([v, g], axis=-1))
        out = compute_swiglu_hidden(act, x)
        np.testing.assert_allclose(np.array(out), 30.0 * np.ones((1, 1, 2)), atol=1e-6)


# ─── compute_effective_dims ──────────────────────────────────────────────────


class TestComputeEffectiveDimsStandard:
    def test_none_uses_input(self):
        hid, out = compute_effective_dims_standard(64, None, None)
        assert hid == 64
        assert out == 64

    def test_custom_dims(self):
        hid, out = compute_effective_dims_standard(64, 256, 32)
        assert hid == 256
        assert out == 32

    def test_partial_none(self):
        hid, out = compute_effective_dims_standard(64, 128, None)
        assert hid == 128
        assert out == 64


class TestComputeEffectiveDimsFused:
    def test_none_uses_scaled_input(self):
        hid, out = compute_effective_dims_fused(768, None, None)
        expected_hid = (int(768 * 2 / 3) + 7) // 8 * 8
        assert hid == expected_hid
        assert out == 768

    def test_rounding_to_8(self):
        hid, _ = compute_effective_dims_fused(64, 100, None)
        assert hid % 8 == 0

    def test_output_falls_back_to_input(self):
        _, out = compute_effective_dims_fused(128, 256, None)
        assert out == 128

    def test_explicit_output(self):
        _, out = compute_effective_dims_fused(128, 256, 64)
        assert out == 64


class TestComputeEffectiveDimsAligned:
    def test_align_to_8(self):
        hid, out = compute_effective_dims_aligned(64, 100, None, 8)
        assert hid % 8 == 0

    def test_align_to_16(self):
        hid, out = compute_effective_dims_aligned(64, 100, None, 16)
        assert hid % 16 == 0

    def test_output_falls_back(self):
        _, out = compute_effective_dims_aligned(64, 256, None, 8)
        assert out == 64

    def test_known_calculation(self):
        d = int(96 * 2 / 3)
        expected = d + (-d % 8)
        hid, _ = compute_effective_dims_aligned(64, 96, None, 8)
        assert hid == expected


# ─── build helpers ───────────────────────────────────────────────────────────


class TestBuildDense:
    def test_returns_dense(self):
        layer = build_dense(128, True, "test_proj")
        assert isinstance(layer, keras.layers.Dense)

    def test_units(self):
        layer = build_dense(256, True, "proj")
        assert layer.units == 256

    def test_no_bias(self):
        layer = build_dense(64, False, "proj")
        assert not layer.use_bias

    def test_name(self):
        layer = build_dense(32, True, "my_name")
        assert layer.name == "my_name"


class TestBuildSwiGLUFFNLayers:
    def test_returns_two_layers(self):
        fused, out = build_swiglu_ffn_layers(64, 32, True)
        assert isinstance(fused, keras.layers.Dense)
        assert isinstance(out, keras.layers.Dense)

    def test_fused_units_doubled(self):
        fused, _ = build_swiglu_ffn_layers(64, 32, True)
        assert fused.units == 128

    def test_output_units(self):
        _, out = build_swiglu_ffn_layers(64, 32, True)
        assert out.units == 32

    def test_fused_name(self):
        fused, _ = build_swiglu_ffn_layers(64, 32, True)
        assert fused.name == "fused_gate_and_value_projection"

    def test_output_name(self):
        _, out = build_swiglu_ffn_layers(64, 32, True)
        assert out.name == "output_projection"


class TestBuildSwiGLUAlignedLayers:
    def test_returns_three_layers(self):
        val, gate, out = build_swiglu_aligned_layers(64, 32, True)
        assert isinstance(val, keras.layers.Dense)
        assert isinstance(gate, keras.layers.Dense)
        assert isinstance(out, keras.layers.Dense)

    def test_names(self):
        val, gate, out = build_swiglu_aligned_layers(64, 32, True)
        assert val.name == "value_projection"
        assert gate.name == "gate_projection"
        assert out.name == "output_projection"

    def test_hidden_units(self):
        val, gate, _ = build_swiglu_aligned_layers(64, 32, True)
        assert val.units == 64
        assert gate.units == 64


# ─── SwiGLUFFN factory ───────────────────────────────────────────────────────


class TestSwiGLUFFN:
    def test_returns_keras_model(self):
        m = make_standard()
        assert isinstance(m, keras.Model)

    def test_output_shape_flattened_default(self):
        m = make_standard(dim=64, hid=256)
        x = make_x(dim=64)
        out = m(x, training=False)
        assert out.shape == (2, 8, 64)

    def test_output_features_override(self):
        m = SwiGLUFFN(input_features=64, hidden_features=256, output_features=32)
        out = m(make_x(dim=64), training=False)
        assert out.shape[-1] == 32

    def test_fused_proj_attribute(self):
        m = make_standard()
        assert hasattr(m, "fused_gate_and_value_projection")
        assert isinstance(m.fused_gate_and_value_projection, keras.layers.Dense)

    def test_output_proj_attribute(self):
        m = make_standard()
        assert hasattr(m, "output_projection")
        assert isinstance(m.output_projection, keras.layers.Dense)

    def test_activation_layer_attribute(self):
        m = make_standard()
        assert hasattr(m, "activation_layer")

    def test_custom_activation_used(self):
        act = keras.layers.Activation("gelu")
        m = SwiGLUFFN(input_features=64, activation_layer=act)
        assert m.activation_layer is act

    def test_default_activation_is_silu(self):
        m = make_standard()
        assert isinstance(m.activation_layer, keras.layers.Activation)

    def test_no_bias(self):
        m = SwiGLUFFN(input_features=64, use_bias=False)
        out = m(make_x(dim=64), training=False)
        assert out.shape[-1] == 64

    def test_fused_proj_built_after_construction(self):
        m = make_standard()
        assert m.fused_gate_and_value_projection.built

    def test_output_deterministic(self):
        m = make_standard()
        x = make_x()
        o1 = np.array(m(x, training=False))
        o2 = np.array(m(x, training=False))
        np.testing.assert_allclose(o1, o2, atol=1e-6)

    def test_training_kwarg_accepted(self):
        m = make_standard()
        out = m(make_x(), training=True)
        assert out.shape == (2, 8, 64)


# ─── SwiGLUFFNFused factory ──────────────────────────────────────────────────


class TestSwiGLUFFNFused:
    def test_returns_keras_model(self):
        m = make_fused()
        assert isinstance(m, keras.Model)

    def test_output_shape(self):
        m = make_fused(dim=64, hid=256)
        out = m(make_x(dim=64), training=False)
        assert out.shape == (2, 8, 64)

    def test_output_features_override(self):
        m = SwiGLUFFNFused(input_features=64, hidden_features=256, output_features=32)
        out = m(make_x(dim=64), training=False)
        assert out.shape[-1] == 32

    def test_fused_proj_attribute(self):
        m = make_fused()
        assert isinstance(m.fused_gate_and_value_projection, keras.layers.Dense)

    def test_output_proj_attribute(self):
        m = make_fused()
        assert isinstance(m.output_projection, keras.layers.Dense)

    def test_activation_always_silu(self):
        act = keras.activations.gelu
        m = SwiGLUFFNFused(input_features=64, activation_layer=act)
        assert isinstance(m.activation_layer, keras.layers.Activation)

    def test_drop_layer_attribute(self):
        m = make_fused()
        assert isinstance(m.drop_layer, keras.layers.Dropout)

    def test_drop_layer_rate(self):
        m = SwiGLUFFNFused(input_features=64, drop_rate=0.3)
        assert m.drop_layer.rate == 0.3

    def test_hidden_dim_rounded(self):
        m = SwiGLUFFNFused(input_features=64, hidden_features=100)
        expected_hid = (int(100 * 2 / 3) + 7) // 8 * 8
        assert m.fused_gate_and_value_projection.units == 2 * expected_hid

    def test_fused_proj_built_after_construction(self):
        m = make_fused()
        assert m.fused_gate_and_value_projection.built

    def test_deterministic_at_inference(self):
        m = make_fused()
        x = make_x()
        o1 = np.array(m(x, training=False))
        o2 = np.array(m(x, training=False))
        np.testing.assert_allclose(o1, o2, atol=1e-6)

    def test_no_dropout_at_inference(self):
        m = SwiGLUFFNFused(input_features=64, drop_rate=0.9)
        x = make_x()
        o1 = np.array(m(x, training=False))
        o2 = np.array(m(x, training=False))
        np.testing.assert_allclose(o1, o2, atol=1e-6)

    def test_custom_name(self):
        m = SwiGLUFFNFused(input_features=64, name="my_mlp")
        assert m.name == "my_mlp"

    def test_no_bias(self):
        m = SwiGLUFFNFused(input_features=64, use_bias=False)
        out = m(make_x(dim=64), training=False)
        assert out.shape[-1] == 64


# ─── SwiGLUFFNAligned factory ────────────────────────────────────────────────


class TestSwiGLUFFNAligned:
    def test_returns_keras_model(self):
        m = make_aligned()
        assert isinstance(m, keras.Model)

    def test_output_shape(self):
        m = make_aligned(dim=64, hid=256)
        out = m(make_x(dim=64), training=False)
        assert out.shape == (2, 8, 64)

    def test_value_projection_attribute(self):
        m = make_aligned()
        assert isinstance(m.value_projection, keras.layers.Dense)

    def test_gate_projection_attribute(self):
        m = make_aligned()
        assert isinstance(m.gate_projection, keras.layers.Dense)

    def test_output_projection_attribute(self):
        m = make_aligned()
        assert isinstance(m.output_projection, keras.layers.Dense)

    def test_hidden_aligned_to_8(self):
        m = make_aligned(dim=64, hid=100, align_to=8)
        d = int(100 * 2 / 3)
        expected = d + (-d % 8)
        assert m.value_projection.units == expected

    def test_align_to_16(self):
        m = make_aligned(dim=64, hid=100, align_to=16)
        assert m.value_projection.units % 16 == 0

    def test_custom_activation(self):
        act = keras.layers.Activation("gelu")
        m = SwiGLUFFNAligned(input_features=64, activation_layer=act)
        assert m.activation_layer is act

    def test_deterministic(self):
        m = make_aligned()
        x = make_x()
        o1 = np.array(m(x, training=False))
        o2 = np.array(m(x, training=False))
        np.testing.assert_allclose(o1, o2, atol=1e-6)

    def test_output_features_override(self):
        m = SwiGLUFFNAligned(input_features=64, hidden_features=128, output_features=32)
        out = m(make_x(dim=64), training=False)
        assert out.shape[-1] == 32


# ─── Weight porting behavior ─────────────────────────────────────────────────


class TestWeightPortingPattern:
    def test_fused_set_weights_after_forward(self):
        m = make_fused(dim=64, hid=256)
        x = make_x(dim=64)
        _ = m(x, training=False)
        hid_dim = (int(256 * 2 / 3) + 7) // 8 * 8
        w = np.ones((64, 2 * hid_dim), dtype="float32") * 0.01
        b = np.zeros(2 * hid_dim, dtype="float32")
        m.fused_gate_and_value_projection.set_weights([w, b])
        np.testing.assert_array_equal(
            m.fused_gate_and_value_projection.get_weights()[1], b
        )

    def test_output_proj_set_weights_after_forward(self):
        m = make_fused(dim=64, hid=256)
        x = make_x(dim=64)
        _ = m(x, training=False)
        hid_dim = (int(256 * 2 / 3) + 7) // 8 * 8
        w = np.ones((hid_dim, 64), dtype="float32") * 0.01
        b = np.zeros(64, dtype="float32")
        m.output_projection.set_weights([w, b])
        np.testing.assert_array_equal(m.output_projection.get_weights()[1], b)

    def test_fused_proj_built_for_set_weights(self):
        m = make_fused()
        assert m.fused_gate_and_value_projection.built

    def test_weight_change_affects_output(self):
        m = make_fused(dim=64, hid=256)
        x = make_x(dim=64)
        _ = m(x, training=False)
        out_before = np.array(m(x, training=False))
        hid_dim = (int(256 * 2 / 3) + 7) // 8 * 8
        w = np.zeros((64, 2 * hid_dim), dtype="float32")
        b = np.zeros(2 * hid_dim, dtype="float32")
        m.fused_gate_and_value_projection.set_weights([w, b])
        out_after = np.array(m(x, training=False))
        assert not np.allclose(out_before, out_after)

    def test_standard_ffn_weight_setting(self):
        m = make_standard(dim=32, hid=64)
        x = make_x(dim=32)
        _ = m(x, training=False)
        w = np.ones((32, 128), dtype="float32") * 0.001
        b = np.zeros(128, dtype="float32")
        m.fused_gate_and_value_projection.set_weights([w, b])
        np.testing.assert_array_equal(
            m.fused_gate_and_value_projection.get_weights()[1], b
        )


# ─── Training / dropout behavior ─────────────────────────────────────────────


class TestDropoutBehavior:
    def test_inference_same_twice(self):
        m = SwiGLUFFNFused(input_features=32, drop_rate=0.5)
        x = make_x(batch=4, seq=8, dim=32)
        o1 = np.array(m(x, training=False))
        o2 = np.array(m(x, training=False))
        np.testing.assert_allclose(o1, o2, atol=1e-6)

    def test_zero_drop_rate_training_equals_inference(self):
        m = SwiGLUFFNFused(input_features=32, drop_rate=0.0)
        x = make_x(dim=32)
        o_train = np.array(m(x, training=True))
        o_infer = np.array(m(x, training=False))
        np.testing.assert_allclose(o_train, o_infer, atol=1e-5)

    def test_standard_ffn_no_dropout(self):
        m = SwiGLUFFN(input_features=32, drop_rate=0.9)
        x = make_x(dim=32)
        o1 = np.array(m(x, training=True))
        o2 = np.array(m(x, training=True))
        np.testing.assert_allclose(o1, o2, atol=1e-6)

    def test_aligned_no_dropout(self):
        m = SwiGLUFFNAligned(input_features=32, drop_rate=0.9)
        x = make_x(dim=32)
        o1 = np.array(m(x, training=True))
        o2 = np.array(m(x, training=True))
        np.testing.assert_allclose(o1, o2, atol=1e-6)


# ─── Integration with Block ──────────────────────────────────────────────────


class TestIntegrationWithBlock:
    def test_block_uses_swiglu_fused(self):
        from paz.models.foundation.dinov2.layers.block import Block

        block = Block(
            dimension=64,
            number_of_heads=4,
            feedforward_network_layer="swiglu",
        )
        assert isinstance(block.mlp, keras.Model)
        x = make_x(batch=1, seq=16, dim=64)
        out = block(x, training=False)
        assert out.shape == (1, 16, 64)

    def test_block_mlp_has_fused_proj(self):
        from paz.models.foundation.dinov2.layers.block import Block

        block = Block(
            dimension=64, number_of_heads=4, feedforward_network_layer="swiglu"
        )
        assert hasattr(block.mlp, "fused_gate_and_value_projection")

    def test_block_mlp_has_output_proj(self):
        from paz.models.foundation.dinov2.layers.block import Block

        block = Block(
            dimension=64, number_of_heads=4, feedforward_network_layer="swiglu"
        )
        assert hasattr(block.mlp, "output_projection")

    def test_block_mlp_weight_access_pattern(self):
        from paz.models.foundation.dinov2.layers.block import Block

        block = Block(
            dimension=64, number_of_heads=4, feedforward_network_layer="swiglu"
        )
        x = make_x(batch=1, seq=16, dim=64)
        _ = block(x, training=False)
        hid = block.mlp.fused_gate_and_value_projection.units
        assert hid > 0

    def test_nested_tensor_block_swiglu_forward(self):
        from paz.models.foundation.dinov2.layers.block import NestedTensorBlock

        block = NestedTensorBlock(
            dimension=64,
            number_of_heads=4,
            feedforward_network_layer="swiglu",
        )
        x = make_x(batch=1, seq=16, dim=64)
        out = block(x, training=False)
        assert out.shape == (1, 16, 64)

    def test_swiglufused_string_alias(self):
        from paz.models.foundation.dinov2.layers.block import Block

        block = Block(
            dimension=64,
            number_of_heads=4,
            feedforward_network_layer="swiglufused",
        )
        assert hasattr(block.mlp, "fused_gate_and_value_projection")

    def test_trainable_variables_exist(self):
        from paz.models.foundation.dinov2.layers.block import Block

        block = Block(
            dimension=64, number_of_heads=4, feedforward_network_layer="swiglu"
        )
        x = make_x(batch=1, seq=16, dim=64)
        _ = block(x, training=False)
        mlp_vars = block.mlp.trainable_variables
        assert len(mlp_vars) > 0
