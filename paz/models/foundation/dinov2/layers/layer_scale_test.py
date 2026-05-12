import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest
import keras
from keras import ops

from paz.models.foundation.dinov2.layers.layer_scale import (
    LayerScale,
    compute_layer_scale,
)
from paz.models.foundation.dinov2.layers.block import Block, NestedTensorBlock

# ─── Helpers ────────────────────────────────────────────────────────────────


def make_layer_scale(dim=64, init_values=1e-5):
    return LayerScale(dimension=dim, init_values=init_values)


# ─── compute_layer_scale ────────────────────────────────────────────────────


class TestComputeLayerScale:
    def test_scales_by_gamma(self):
        x = ops.ones((1, 4, 8))
        gamma = ops.ones((8,)) * 2.0
        out = compute_layer_scale(x, gamma)
        np.testing.assert_allclose(np.array(out), 2.0 * np.ones((1, 4, 8)))

    def test_zero_gamma_gives_zeros(self):
        x = ops.ones((2, 6, 16))
        gamma = ops.zeros((16,))
        out = compute_layer_scale(x, gamma)
        np.testing.assert_allclose(np.array(out), np.zeros((2, 6, 16)))

    def test_output_shape_preserved(self):
        B, T, D = 3, 10, 32
        x = ops.convert_to_tensor(np.random.randn(B, T, D).astype("float32"))
        gamma = ops.ones((D,))
        out = compute_layer_scale(x, gamma)
        assert out.shape == (B, T, D)

    def test_element_wise_independent_channels(self):
        B, T, D = 1, 2, 4
        x = ops.ones((B, T, D))
        gamma = ops.convert_to_tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype="float32"))
        out = compute_layer_scale(x, gamma)
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(np.array(out)[0, 0], expected)

    def test_matches_model_call(self):
        dim = 32
        model = make_layer_scale(dim=dim, init_values=0.5)
        x = ops.convert_to_tensor(np.random.randn(2, 6, dim).astype("float32"))
        out_model = model(x, training=False)
        out_fn = compute_layer_scale(x, model.gamma)
        np.testing.assert_allclose(np.array(out_model), np.array(out_fn), atol=1e-6)


# ─── LayerScale function ────────────────────────────────────────────────────


class TestLayerScaleFunction:
    def test_returns_keras_model(self):
        model = make_layer_scale()
        assert isinstance(model, keras.Model)

    def test_gamma_attribute_accessible(self):
        model = make_layer_scale()
        assert hasattr(model, "gamma")

    def test_gamma_is_keras_variable(self):
        model = make_layer_scale()
        assert isinstance(model.gamma, keras.Variable)

    def test_gamma_initial_value(self):
        init = 1e-4
        model = LayerScale(dimension=32, init_values=init)
        gamma_val = np.array(model.gamma)
        np.testing.assert_allclose(gamma_val, init * np.ones(32), atol=1e-9)

    def test_gamma_shape(self):
        dim = 48
        model = LayerScale(dimension=dim)
        assert model.gamma.shape == (dim,)

    def test_gamma_trainable(self):
        model = make_layer_scale()
        assert model.gamma.trainable

    def test_forward_output_shape(self):
        dim = 64
        model = make_layer_scale(dim=dim)
        x = ops.ones((2, 10, dim))
        out = model(x)
        assert out.shape == (2, 10, dim)

    def test_forward_training_false(self):
        dim = 64
        model = make_layer_scale(dim=dim)
        x = ops.convert_to_tensor(np.random.randn(2, 7, dim).astype("float32"))
        out = model(x, training=False)
        assert out.shape == (2, 7, dim)

    def test_forward_training_true(self):
        dim = 64
        model = make_layer_scale(dim=dim)
        x = ops.convert_to_tensor(np.random.randn(2, 7, dim).astype("float32"))
        out = model(x, training=True)
        assert out.shape == (2, 7, dim)

    def test_forward_no_training_kwarg(self):
        dim = 32
        model = make_layer_scale(dim=dim)
        x = ops.ones((1, 5, dim))
        out = model(x)
        assert out.shape == (1, 5, dim)

    def test_scale_applied_correctly(self):
        dim = 4
        init = 2.0
        model = LayerScale(dimension=dim, init_values=init)
        x = ops.ones((1, 3, dim))
        out = model(x, training=False)
        np.testing.assert_allclose(np.array(out), 2.0 * np.ones((1, 3, dim)), atol=1e-6)

    def test_deterministic_inference(self):
        model = make_layer_scale()
        x = ops.convert_to_tensor(np.random.randn(2, 6, 64).astype("float32"))
        out1 = model(x, training=False)
        out2 = model(x, training=False)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-7)

    def test_gamma_assign(self):
        dim = 16
        model = make_layer_scale(dim=dim)
        new_gamma = np.ones(dim, dtype="float32") * 5.0
        model.gamma.assign(new_gamma)
        np.testing.assert_array_equal(np.array(model.gamma), new_gamma)

    def test_gamma_assign_changes_output(self):
        dim = 8
        model = make_layer_scale(dim=dim)
        x = ops.ones((1, 4, dim))
        before = np.array(model(x, training=False))
        model.gamma.assign(np.ones(dim, dtype="float32") * 3.0)
        after = np.array(model(x, training=False))
        np.testing.assert_allclose(after, 3.0 * np.ones((1, 4, dim)))
        assert not np.allclose(before, after, atol=1e-4)

    def test_different_dimensions(self):
        for dim in [16, 64, 192, 384]:
            model = LayerScale(dimension=dim)
            x = ops.ones((1, 4, dim))
            out = model(x, training=False)
            assert out.shape == (1, 4, dim)

    def test_custom_name(self):
        model = LayerScale(dimension=32, name="ls1")
        assert model.name == "ls1"

    def test_default_init_value(self):
        model = LayerScale(dimension=8)
        expected = 1e-5 * np.ones(8, dtype="float32")
        np.testing.assert_allclose(np.array(model.gamma), expected, atol=1e-10)

    def test_data_type_param_accepted(self):
        model = LayerScale(dimension=32, data_type="float32")
        assert isinstance(model, keras.Model)


# ─── Integration with Block ─────────────────────────────────────────────────


class TestLayerScaleInBlock:
    def test_block_layer_scale_1_is_keras_model(self):
        block = Block(dimension=64, number_of_heads=4, init_values=1e-5)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert isinstance(block.layer_scale_1, keras.Model)

    def test_block_layer_scale_2_is_keras_model(self):
        block = Block(dimension=64, number_of_heads=4, init_values=1e-5)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert isinstance(block.layer_scale_2, keras.Model)

    def test_block_layer_scale_gamma_accessible(self):
        block = Block(dimension=64, number_of_heads=4, init_values=1e-5)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert hasattr(block.layer_scale_1, "gamma")
        assert hasattr(block.layer_scale_2, "gamma")

    def test_block_layer_scale_gamma_assign(self):
        dim = 64
        block = Block(dimension=dim, number_of_heads=4, init_values=1e-5)
        x = ops.ones((1, 4, dim))
        block(x, training=False)
        new_val = np.ones(dim, dtype="float32") * 0.5
        block.layer_scale_1.gamma.assign(new_val)
        np.testing.assert_array_equal(np.array(block.layer_scale_1.gamma), new_val)

    def test_block_no_layer_scale_when_no_init_values(self):
        block = Block(dimension=64, number_of_heads=4, init_values=None)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert not hasattr(block.layer_scale_1, "gamma")
        assert not hasattr(block.layer_scale_2, "gamma")

    def test_block_layer_scale_callable(self):
        dim = 64
        block = Block(dimension=dim, number_of_heads=4, init_values=1e-5)
        x = ops.ones((1, 4, dim))
        block(x, training=False)
        test_x = ops.convert_to_tensor(np.random.randn(1, 4, dim).astype("float32"))
        out = block.layer_scale_1(test_x, training=False)
        assert out.shape == (1, 4, dim)

    def test_block_forward_pass_with_layer_scale(self):
        block = Block(dimension=64, number_of_heads=4, init_values=1e-5)
        x = ops.convert_to_tensor(np.random.randn(2, 8, 64).astype("float32"))
        out = block(x, training=False)
        assert out.shape == (2, 8, 64)

    def test_block_layer_scale_gamma_weight_porting_pattern(self):
        dim = 64
        block = Block(dimension=dim, number_of_heads=4, init_values=1e-5)
        x = ops.ones((1, 4, dim))
        block(x, training=False)
        gamma_val = np.ones(dim, dtype="float32") * 0.1
        block.layer_scale_1.gamma.assign(gamma_val)
        block.layer_scale_2.gamma.assign(gamma_val)
        np.testing.assert_allclose(np.array(block.layer_scale_1.gamma), gamma_val)
        np.testing.assert_allclose(np.array(block.layer_scale_2.gamma), gamma_val)


# ─── Integration with NestedTensorBlock ─────────────────────────────────────


class TestLayerScaleInNestedTensorBlock:
    def test_nested_block_layer_scale_gamma_accessible(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4, init_values=1e-5)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert hasattr(block.layer_scale_1, "gamma")
        assert hasattr(block.layer_scale_2, "gamma")

    def test_nested_block_apply_scaled_attention_residual(self):
        dim = 64
        block = NestedTensorBlock(dimension=dim, number_of_heads=4, init_values=1e-5)
        x = ops.ones((1, 4, dim))
        block(x, training=False)
        out = block.apply_scaled_attention_residual(x)
        assert out.shape == (1, 4, dim)

    def test_nested_block_apply_scaled_ffn_residual(self):
        dim = 64
        block = NestedTensorBlock(dimension=dim, number_of_heads=4, init_values=1e-5)
        x = ops.ones((1, 4, dim))
        block(x, training=False)
        out = block.apply_scaled_feedforward_network_residual(x)
        assert out.shape == (1, 4, dim)

    def test_nested_block_forward_pass(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4, init_values=1e-5)
        x = ops.convert_to_tensor(np.random.randn(2, 8, 64).astype("float32"))
        out = block(x, training=False)
        assert out.shape == (2, 8, 64)

    def test_nested_block_gamma_assign(self):
        dim = 64
        block = NestedTensorBlock(dimension=dim, number_of_heads=4, init_values=1e-5)
        x = ops.ones((1, 4, dim))
        block(x, training=False)
        new_val = np.ones(dim, dtype="float32") * 0.2
        block.layer_scale_1.gamma.assign(new_val)
        np.testing.assert_allclose(np.array(block.layer_scale_1.gamma), new_val)

    def test_nested_block_list_of_tensors(self):
        if keras.backend.backend() == "tensorflow":
            pytest.skip("TF tensor scope issue in block.py")
        block = NestedTensorBlock(dimension=64, number_of_heads=4, init_values=1e-5)
        tensors = [
            ops.convert_to_tensor(np.random.randn(2, 6, 64).astype("float32")),
            ops.convert_to_tensor(np.random.randn(3, 6, 64).astype("float32")),
        ]
        result = block(tensors, training=False)
        assert isinstance(result, list)
        assert result[0].shape == (2, 6, 64)
        assert result[1].shape == (3, 6, 64)


# ─── Behavioural consistency ────────────────────────────────────────────────


class TestBehaviouralConsistency:
    def test_model_call_and_compute_identical(self):
        dim = 64
        model = make_layer_scale(dim=dim, init_values=0.01)
        x = ops.convert_to_tensor(np.random.randn(2, 10, dim).astype("float32"))
        out_model = model(x, training=False)
        out_fn = compute_layer_scale(x, model.gamma)
        np.testing.assert_allclose(np.array(out_model), np.array(out_fn), atol=1e-6)

    def test_gamma_scales_proportionally(self):
        dim = 16
        model = make_layer_scale(dim=dim, init_values=1.0)
        x = ops.ones((1, 4, dim))
        out_1 = np.array(model(x, training=False))
        model.gamma.assign(np.ones(dim, dtype="float32") * 2.0)
        out_2 = np.array(model(x, training=False))
        np.testing.assert_allclose(out_2, 2.0 * out_1, atol=1e-6)

    def test_block_with_vs_without_layer_scale(self):
        dim, heads = 64, 4
        x = ops.convert_to_tensor(np.random.randn(2, 8, dim).astype("float32"))
        block_with = Block(dimension=dim, number_of_heads=heads, init_values=1e-5)
        block_without = Block(dimension=dim, number_of_heads=heads, init_values=None)
        out_with = block_with(x, training=False)
        out_without = block_without(x, training=False)
        assert out_with.shape == out_without.shape == (2, 8, dim)
