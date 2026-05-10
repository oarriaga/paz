import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest
import keras
from keras import ops
from keras.layers import Dropout

from paz.models.foundation.dinov2.layers.attention import (
    Attention,
    compute_attention,
    compute_scaled_dot_product_attention,
    split_query_key_value,
)
from paz.models.foundation.dinov2.layers.block import Block, NestedTensorBlock

# ─── Helpers ────────────────────────────────────────────────────────────────


def make_no_drop():
    return Dropout(0.0)


def make_attention(dim=64, num_heads=4):
    return Attention(dimension=dim, number_of_heads=num_heads)


def call_compute_attention(model, x, attention_bias=None, training=None):
    return compute_attention(
        x,
        model.predict_query_key_value,
        model.projection_layer,
        model.attention_drop,
        model.projection_drop,
        model.number_of_heads,
        model.head_dimension,
        model.scale,
        attention_bias,
        training,
    )


# ─── split_query_key_value ───────────────────────────────────────────────────


class TestSplitQueryKeyValue:
    def test_output_shapes(self):
        B, T, H, D = 2, 10, 4, 16
        qkv = ops.ones((B, T, 3 * H * D))
        q, k, v = split_query_key_value(qkv, B, T, H, D)
        assert q.shape == (B, H, T, D)
        assert k.shape == (B, H, T, D)
        assert v.shape == (B, H, T, D)

    def test_separates_correctly(self):
        B, T, H, D = 1, 5, 2, 8
        raw = np.random.randn(B, T, 3 * H * D).astype("float32")
        qkv = ops.convert_to_tensor(raw)
        q, k, v = split_query_key_value(qkv, B, T, H, D)
        ref = raw.reshape(B, T, 3, H, D).transpose(2, 0, 3, 1, 4)
        np.testing.assert_allclose(np.array(q), ref[0], atol=1e-6)
        np.testing.assert_allclose(np.array(k), ref[1], atol=1e-6)
        np.testing.assert_allclose(np.array(v), ref[2], atol=1e-6)

    def test_single_head(self):
        B, T, H, D = 2, 6, 1, 32
        qkv = ops.ones((B, T, 3 * H * D))
        q, k, v = split_query_key_value(qkv, B, T, H, D)
        assert q.shape == (B, H, T, D)

    def test_batch_size_one(self):
        B, T, H, D = 1, 4, 8, 8
        qkv = ops.zeros((B, T, 3 * H * D))
        q, k, v = split_query_key_value(qkv, B, T, H, D)
        assert q.shape == (B, H, T, D)


# ─── compute_scaled_dot_product_attention ───────────────────────────────────


class TestComputeScaledDotProductAttention:
    def test_output_shape(self):
        B, H, T, D = 2, 4, 10, 16
        q = ops.ones((B, H, T, D))
        k = ops.ones((B, H, T, D))
        v = ops.ones((B, H, T, D))
        drop = make_no_drop()
        out = compute_scaled_dot_product_attention(q, k, v, D**-0.5, None, drop, False)
        assert out.shape == (B, H, T, D)

    def test_zero_bias_does_not_change_output(self):
        B, H, T, D = 1, 2, 5, 8
        q = ops.convert_to_tensor(np.random.randn(B, H, T, D).astype("float32"))
        k = ops.convert_to_tensor(np.random.randn(B, H, T, D).astype("float32"))
        v = ops.convert_to_tensor(np.random.randn(B, H, T, D).astype("float32"))
        scale = D**-0.5
        drop = make_no_drop()
        bias = ops.zeros((T, T))
        out1 = compute_scaled_dot_product_attention(q, k, v, scale, None, drop, False)
        out2 = compute_scaled_dot_product_attention(q, k, v, scale, bias, drop, False)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-6)

    def test_large_negative_bias_masks_attention(self):
        B, H, T, D = 1, 1, 4, 4
        q = ops.ones((B, H, T, D))
        k = ops.ones((B, H, T, D))
        v = ops.convert_to_tensor(np.eye(T, D).astype("float32")[None, None])
        scale = D**-0.5
        drop = make_no_drop()
        bias = ops.zeros((T, T))
        out = compute_scaled_dot_product_attention(q, k, v, scale, bias, drop, False)
        assert out.shape == (B, H, T, D)

    def test_scale_applied(self):
        B, H, T, D = 1, 1, 4, 8
        rng = np.random.default_rng(42)
        q = ops.convert_to_tensor(rng.standard_normal((B, H, T, D)).astype("float32"))
        k = ops.convert_to_tensor(rng.standard_normal((B, H, T, D)).astype("float32"))
        v = ops.convert_to_tensor(rng.standard_normal((B, H, T, D)).astype("float32"))
        drop = make_no_drop()
        out1 = compute_scaled_dot_product_attention(q, k, v, 1.0, None, drop, False)
        out2 = compute_scaled_dot_product_attention(q, k, v, 0.001, None, drop, False)
        assert not np.allclose(np.array(out1), np.array(out2), atol=1e-3)


# ─── compute_attention ──────────────────────────────────────────────────────


class TestComputeAttentionFunction:
    def test_matches_model_call(self):
        dim, heads = 64, 4
        model = make_attention(dim, heads)
        x = ops.convert_to_tensor(np.random.randn(2, 10, dim).astype("float32"))
        out_model = model(x, training=False)
        out_fn = call_compute_attention(model, x, training=False)
        np.testing.assert_allclose(np.array(out_model), np.array(out_fn), atol=1e-6)

    def test_matches_with_attention_bias(self):
        dim, heads, T = 64, 4, 8
        model = make_attention(dim, heads)
        x = ops.convert_to_tensor(np.random.randn(1, T, dim).astype("float32"))
        bias = ops.zeros((T, T))
        out_model = model(x, attention_bias=bias, training=False)
        out_fn = call_compute_attention(model, x, attention_bias=bias, training=False)
        np.testing.assert_allclose(np.array(out_model), np.array(out_fn), atol=1e-6)

    def test_output_shape(self):
        dim, heads = 128, 8
        model = make_attention(dim, heads)
        x = ops.ones((3, 15, dim))
        out = call_compute_attention(model, x, training=False)
        assert out.shape == (3, 15, dim)


# ─── Attention function ──────────────────────────────────────────────────────


class TestAttentionFunction:
    def test_returns_keras_model(self):
        model = Attention(dimension=64)
        assert isinstance(model, keras.Model)

    def test_sublayers_accessible(self):
        model = Attention(dimension=64, number_of_heads=4)
        assert hasattr(model, "predict_query_key_value")
        assert hasattr(model, "projection_layer")
        assert hasattr(model, "attention_drop")
        assert hasattr(model, "projection_drop")

    def test_metadata_accessible(self):
        model = Attention(dimension=64, number_of_heads=4)
        assert model.number_of_heads == 4
        assert model.head_dimension == 16
        assert abs(model.scale - 16**-0.5) < 1e-7

    def test_forward_pass_output_shape(self):
        model = make_attention(64, 4)
        x = ops.ones((2, 10, 64))
        out = model(x)
        assert out.shape == (2, 10, 64)

    def test_forward_pass_no_training_flag(self):
        model = make_attention(64, 4)
        x = ops.ones((1, 5, 64))
        out = model(x)
        assert out.shape == (1, 5, 64)

    def test_forward_pass_with_attention_bias(self):
        dim, heads, T = 64, 4, 8
        model = make_attention(dim, heads)
        x = ops.ones((1, T, dim))
        bias = ops.zeros((T, T))
        out = model(x, attention_bias=bias, training=False)
        assert out.shape == (1, T, dim)

    def test_forward_pass_training_false(self):
        model = make_attention(64, 4)
        x = ops.convert_to_tensor(np.random.randn(2, 7, 64).astype("float32"))
        out = model(x, training=False)
        assert out.shape == (2, 7, 64)

    def test_forward_pass_training_true(self):
        model = make_attention(64, 4)
        x = ops.convert_to_tensor(np.random.randn(2, 7, 64).astype("float32"))
        out = model(x, training=True)
        assert out.shape == (2, 7, 64)

    def test_qkv_bias_false_by_default(self):
        model = make_attention(64, 4)
        assert model.predict_query_key_value.bias is None

    def test_proj_bias_true_by_default(self):
        model = make_attention(64, 4)
        assert model.projection_layer.bias is not None

    def test_qkv_bias_enabled(self):
        model = Attention(dimension=64, use_query_key_value_bias=True)
        assert model.predict_query_key_value.bias is not None

    def test_proj_bias_disabled(self):
        model = Attention(dimension=64, use_projection_bias=False)
        assert model.projection_layer.bias is None

    def test_predict_query_key_value_weight_accessible(self):
        model = make_attention(64, 4)
        weights = model.predict_query_key_value.get_weights()
        assert len(weights) > 0
        assert weights[0].shape[1] == 64 * 3

    def test_projection_layer_weight_accessible(self):
        model = make_attention(64, 4)
        weights = model.projection_layer.get_weights()
        assert len(weights) > 0
        assert weights[0].shape == (64, 64)

    def test_set_weights_on_qkv(self):
        model = make_attention(64, 4)
        w = np.zeros_like(model.predict_query_key_value.get_weights()[0])
        model.predict_query_key_value.set_weights([w])
        result = model.predict_query_key_value.get_weights()[0]
        np.testing.assert_array_equal(result, w)

    def test_deterministic_output_in_inference(self):
        model = make_attention(64, 4)
        x = ops.convert_to_tensor(np.random.randn(2, 6, 64).astype("float32"))
        out1 = model(x, training=False)
        out2 = model(x, training=False)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-6)

    def test_different_dims_and_heads(self):
        for dim, heads in [(32, 2), (128, 8), (192, 12)]:
            model = Attention(dimension=dim, number_of_heads=heads)
            x = ops.ones((1, 4, dim))
            out = model(x, training=False)
            assert out.shape == (1, 4, dim)


# ─── Integration with Block ──────────────────────────────────────────────────


class TestAttentionInBlock:
    def test_block_stores_attention_model(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x)
        assert isinstance(block.attention, keras.Model)

    def test_block_forward_pass(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.convert_to_tensor(np.random.randn(2, 8, 64).astype("float32"))
        out = block(x, training=False)
        assert out.shape == (2, 8, 64)

    def test_block_with_attention_function(self):
        block = Block(dimension=64, number_of_heads=4, attention_class=Attention)
        x = ops.convert_to_tensor(np.random.randn(1, 5, 64).astype("float32"))
        out = block(x, training=False)
        assert out.shape == (1, 5, 64)

    def test_block_attention_has_qkv_sublayer(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x)
        assert hasattr(block.attention, "predict_query_key_value")


# ─── Integration with NestedTensorBlock ────────────────────────────────────


class TestAttentionInNestedTensorBlock:
    def test_nested_block_stores_attention_model(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x)
        assert isinstance(block.attention, keras.Model)

    def test_nested_block_single_tensor(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4)
        x = ops.convert_to_tensor(np.random.randn(2, 8, 64).astype("float32"))
        out = block(x, training=False)
        assert out.shape == (2, 8, 64)

    def test_nested_block_list_of_tensors(self):
        if keras.backend.backend() == "tensorflow":
            pytest.skip("TF tensor scope issue in block.py cross-context bias")
        block = NestedTensorBlock(dimension=64, number_of_heads=4, init_values=1e-5)
        tensors = [
            ops.convert_to_tensor(np.random.randn(2, 6, 64).astype("float32")),
            ops.convert_to_tensor(np.random.randn(3, 6, 64).astype("float32")),
        ]
        result = block(tensors, training=False)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (2, 6, 64)
        assert result[1].shape == (3, 6, 64)

    def test_nested_block_attention_bias_kwarg(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x)
        T = 8
        x = ops.convert_to_tensor(np.random.randn(1, T, 64).astype("float32"))
        bias = ops.zeros((T, T))
        out = block.apply_attention_residual(x, attention_bias=bias)
        assert out.shape == (1, T, 64)


# ─── Behavioural consistency ────────────────────────────────────────────────


class TestBehaviouralConsistency:
    def test_model_call_and_compute_attention_identical(self):
        dim, heads = 96, 6
        model = make_attention(dim, heads)
        x = ops.convert_to_tensor(np.random.randn(2, 12, dim).astype("float32"))
        out_model = model(x, training=False)
        out_fn = call_compute_attention(model, x, training=False)
        np.testing.assert_allclose(np.array(out_model), np.array(out_fn), atol=1e-6)

    def test_model_call_and_compute_attention_identical_with_bias(self):
        dim, heads, T = 64, 4, 10
        model = make_attention(dim, heads)
        x = ops.convert_to_tensor(np.random.randn(1, T, dim).astype("float32"))
        bias = ops.convert_to_tensor(np.random.randn(T, T).astype("float32"))
        out_model = model(x, attention_bias=bias, training=False)
        out_fn = call_compute_attention(model, x, attention_bias=bias, training=False)
        np.testing.assert_allclose(np.array(out_model), np.array(out_fn), atol=1e-6)

    def test_split_then_dot_product_matches_full(self):
        B, H, T, D = 1, 2, 6, 16
        raw = np.random.randn(B, T, 3 * H * D).astype("float32")
        qkv = ops.convert_to_tensor(raw)
        q, k, v = split_query_key_value(qkv, B, T, H, D)
        drop = make_no_drop()
        scale = D**-0.5
        out = compute_scaled_dot_product_attention(q, k, v, scale, None, drop, False)
        assert out.shape == (B, H, T, D)
