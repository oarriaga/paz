import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest
import keras
from keras import ops

from paz.models.foundation.dinov2.layers.drop_path import (
    DropPath,
    bernoulli_random,
    dropout_rate,
)
from paz.models.foundation.dinov2.layers.block import Block, NestedTensorBlock

# ─── Helpers ────────────────────────────────────────────────────────────────


def make_x(batch=2, seq=8, dim=64):
    return ops.convert_to_tensor(np.random.randn(batch, seq, dim).astype("float32"))


# ─── bernoulli_random ────────────────────────────────────────────────────────


class TestBernoulliRandom:
    def test_output_shape(self):
        out = bernoulli_random(shape=(4, 1, 1), probabilities=0.9)
        assert out.shape == (4, 1, 1)

    def test_binary_values(self):
        out = np.array(bernoulli_random(shape=(100,), probabilities=0.5))
        unique = set(out.tolist())
        assert unique.issubset({0.0, 1.0})

    def test_probability_zero_gives_all_zeros(self):
        out = bernoulli_random(shape=(50,), probabilities=0.0)
        np.testing.assert_array_equal(np.array(out), np.zeros(50))

    def test_probability_one_gives_all_ones(self):
        out = bernoulli_random(shape=(50,), probabilities=1.0)
        np.testing.assert_array_equal(np.array(out), np.ones(50))

    def test_dtype_propagated(self):
        out = bernoulli_random(shape=(4,), probabilities=0.5, data_type="float32")
        assert np.array(out).dtype == np.float32


# ─── dropout_rate ────────────────────────────────────────────────────────────


class TestDropoutRate:
    def test_identity_when_training_false(self):
        x = make_x()
        out = dropout_rate(x, drop_probability=0.5, training=False)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_identity_when_prob_zero(self):
        x = make_x()
        out = dropout_rate(x, drop_probability=0.0, training=True)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_identity_default_args(self):
        x = make_x()
        out = dropout_rate(x)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_all_zeros_when_prob_one(self):
        x = make_x()
        out = dropout_rate(x, drop_probability=1.0, training=True)
        np.testing.assert_array_equal(np.array(out), np.zeros_like(np.array(x)))

    def test_output_shape_preserved(self):
        x = make_x(batch=3, seq=10, dim=32)
        out = dropout_rate(x, drop_probability=0.5, training=True)
        assert out.shape == x.shape

    def test_training_true_may_change_output(self):
        x = ops.ones((16, 8, 64))
        out = dropout_rate(x, drop_probability=0.5, training=True)
        assert not np.allclose(np.array(out), np.array(x), atol=1e-4)

    def test_mask_is_per_sample(self):
        x = ops.ones((4, 8, 64))
        out = np.array(dropout_rate(x, drop_probability=0.5, training=True))
        for b in range(4):
            sample = out[b]
            first_row = sample[0]
            for row in sample:
                np.testing.assert_allclose(row, first_row, atol=1e-6)


# ─── DropPath factory ────────────────────────────────────────────────────────


class TestDropPathFunction:
    def test_returns_keras_model(self):
        model = DropPath(drop_probability=0.1)
        assert isinstance(model, keras.Model)

    def test_named_model(self):
        model = DropPath(drop_probability=0.1, name="dp_test")
        assert model.name == "dp_test"

    def test_forward_shape_training_false(self):
        model = DropPath(drop_probability=0.3)
        x = make_x()
        out = model(x, training=False)
        assert out.shape == x.shape

    def test_forward_shape_training_true(self):
        model = DropPath(drop_probability=0.3)
        x = make_x()
        out = model(x, training=True)
        assert out.shape == x.shape

    def test_identity_at_inference(self):
        model = DropPath(drop_probability=0.5)
        x = make_x()
        out = model(x, training=False)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_identity_when_prob_zero(self):
        model = DropPath(drop_probability=0.0)
        x = make_x()
        out = model(x, training=True)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_zeros_when_prob_one_training_true(self):
        model = DropPath(drop_probability=1.0)
        x = make_x()
        out = model(x, training=True)
        np.testing.assert_array_equal(np.array(out), np.zeros_like(np.array(x)))

    def test_no_kwarg_call(self):
        model = DropPath(drop_probability=0.1)
        x = make_x()
        out = model(x)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_none_prob_treated_as_zero(self):
        model = DropPath(drop_probability=None)
        x = make_x()
        out = model(x, training=True)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_no_trainable_variables(self):
        model = DropPath(drop_probability=0.3)
        assert len(model.trainable_variables) == 0

    def test_stochastic_at_training_large_batch(self):
        model = DropPath(drop_probability=0.5)
        x = ops.ones((32, 8, 64))
        out = model(x, training=True)
        assert not np.allclose(np.array(out), np.array(x), atol=1e-4)

    def test_different_drop_probs_inference_identity(self):
        for prob in [0.1, 0.3, 0.5, 0.9]:
            model = DropPath(drop_probability=prob)
            x = make_x()
            out = model(x, training=False)
            np.testing.assert_array_equal(np.array(out), np.array(x))


# ─── Integration with Block ──────────────────────────────────────────────────


class TestDropPathInBlock:
    def test_block_drop_path1_is_keras_model_when_nonzero(self):
        block = Block(dimension=64, number_of_heads=4, drop_path=0.1)
        x = make_x()
        block(x, training=False)
        assert isinstance(block.drop_path1, keras.Model)

    def test_block_drop_path2_is_keras_model_when_nonzero(self):
        block = Block(dimension=64, number_of_heads=4, drop_path=0.1)
        x = make_x()
        block(x, training=False)
        assert isinstance(block.drop_path2, keras.Model)

    def test_block_drop_path_is_identity_when_zero(self):
        block = Block(dimension=64, number_of_heads=4, drop_path=0.0)
        x = make_x()
        block(x, training=False)
        assert isinstance(block.drop_path1, keras.layers.Identity)
        assert isinstance(block.drop_path2, keras.layers.Identity)

    def test_block_forward_training_false(self):
        block = Block(dimension=64, number_of_heads=4, drop_path=0.1)
        x = make_x()
        out = block(x, training=False)
        assert out.shape == x.shape

    def test_block_forward_training_true(self):
        block = Block(dimension=64, number_of_heads=4, drop_path=0.1)
        x = make_x()
        out = block(x, training=True)
        assert out.shape == x.shape

    def test_block_forward_no_drop_path(self):
        block = Block(dimension=64, number_of_heads=4, drop_path=0.0)
        x = make_x()
        out = block(x, training=False)
        assert out.shape == x.shape

    def test_block_drop_path_callable_inference(self):
        block = Block(dimension=64, number_of_heads=4, drop_path=0.1)
        x = make_x()
        block(x, training=False)
        out = block.drop_path1(x, training=False)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_block_drop_path_zeros_when_prob_one(self):
        block = Block(dimension=64, number_of_heads=4, drop_path=1.0)
        x = make_x()
        block(x, training=False)
        out = block.drop_path1(x, training=True)
        np.testing.assert_array_equal(np.array(out), np.zeros_like(np.array(x)))

    def test_block_sample_drop_ratio_stored(self):
        block = Block(dimension=64, number_of_heads=4, drop_path=0.3)
        assert block.sample_drop_ratio == 0.3


# ─── Integration with NestedTensorBlock ──────────────────────────────────────


class TestDropPathInNestedTensorBlock:
    def test_nested_block_drop_path1_is_keras_model(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4, drop_path=0.1)
        x = make_x()
        block(x, training=False)
        assert isinstance(block.drop_path1, keras.Model)

    def test_nested_block_forward_tensor(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4, drop_path=0.1)
        x = make_x()
        out = block(x, training=False)
        assert out.shape == x.shape

    def test_nested_block_forward_list(self):
        if keras.backend.backend() == "tensorflow":
            pytest.skip("TF tensor scope issue in block.py")
        block = NestedTensorBlock(dimension=64, number_of_heads=4, drop_path=0.1)
        tensors = [make_x(batch=2), make_x(batch=3)]
        result = block(tensors, training=False)
        assert isinstance(result, list)
        assert result[0].shape == (2, 8, 64)
        assert result[1].shape == (3, 8, 64)

    def test_nested_block_forward_training_true(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4, drop_path=0.1)
        x = make_x()
        out = block(x, training=True)
        assert out.shape == x.shape


# ─── Behavioural consistency ──────────────────────────────────────────────────


class TestBehaviouralConsistency:
    def test_drop_path_inference_matches_dropout_rate(self):
        model = DropPath(drop_probability=0.5)
        x = make_x()
        out_model = np.array(model(x, training=False))
        out_fn = np.array(dropout_rate(x, drop_probability=0.5, training=False))
        np.testing.assert_array_equal(out_model, out_fn)

    def test_block_inference_deterministic(self):
        block = Block(dimension=64, number_of_heads=4, drop_path=0.3)
        x = make_x()
        out1 = np.array(block(x, training=False))
        out2 = np.array(block(x, training=False))
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_block_drop_path_bypassed_at_inference(self):
        block1 = Block(dimension=64, number_of_heads=4, drop_path=0.0)
        block2 = Block(dimension=64, number_of_heads=4, drop_path=0.9)
        x = make_x()
        for w1, w2 in zip(block1.trainable_variables, block2.trainable_variables):
            w2.assign(w1)
        out1 = np.array(block1(x, training=False))
        out2 = np.array(block2(x, training=False))
        np.testing.assert_allclose(out1, out2, atol=1e-5)
