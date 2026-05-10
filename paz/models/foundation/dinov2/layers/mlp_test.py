import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest
import keras
from keras import ops
from keras.layers import Activation

from paz.models.foundation.dinov2.layers.mlp import MLP, compute_mlp
from paz.models.foundation.dinov2.layers.block import Block, NestedTensorBlock

# ─── Helpers ────────────────────────────────────────────────────────────────


def make_no_drop():
    return keras.layers.Dropout(0.0)


def make_mlp(in_features=64, hidden=None, out=None):
    return MLP(input_features=in_features, hidden_features=hidden, output_features=out)


def call_compute_mlp(model, x, training=None):
    return compute_mlp(
        x,
        model.fully_connected_layer_1,
        model.activation,
        model.fully_connected_layer_2,
        model.drop_layer,
        training,
    )


# ─── compute_mlp ────────────────────────────────────────────────────────────


class TestComputeMlp:
    def test_output_shape_default_features(self):
        in_f = 64
        model = make_mlp(in_f)
        x = ops.ones((2, 10, in_f))
        out = call_compute_mlp(model, x, training=False)
        assert out.shape == (2, 10, in_f)

    def test_output_shape_custom_hidden(self):
        in_f, hidden = 32, 128
        model = make_mlp(in_f, hidden=hidden)
        x = ops.ones((1, 5, in_f))
        out = call_compute_mlp(model, x, training=False)
        assert out.shape == (1, 5, in_f)

    def test_output_shape_custom_out(self):
        in_f, out_f = 64, 32
        model = make_mlp(in_f, out=out_f)
        x = ops.ones((1, 4, in_f))
        out = call_compute_mlp(model, x, training=False)
        assert out.shape == (1, 4, out_f)

    def test_matches_model_call(self):
        model = make_mlp(64)
        x = ops.convert_to_tensor(np.random.randn(2, 8, 64).astype("float32"))
        out_model = model(x, training=False)
        out_fn = call_compute_mlp(model, x, training=False)
        np.testing.assert_allclose(np.array(out_model), np.array(out_fn), atol=1e-6)

    def test_activation_applied(self):
        in_f = 16
        fc1 = keras.layers.Dense(in_f, use_bias=False)
        fc2 = keras.layers.Dense(in_f, use_bias=False)
        identity = keras.layers.Activation("linear")
        gelu = keras.layers.Activation("gelu")
        drop = make_no_drop()
        x = ops.ones((1, 4, in_f))
        out_id = compute_mlp(x, fc1, identity, fc2, drop, False)
        out_gelu = compute_mlp(x, fc1, gelu, fc2, drop, False)
        assert not np.allclose(np.array(out_id), np.array(out_gelu), atol=1e-3)

    def test_drop_applied_during_training(self):
        in_f = 64
        model = MLP(input_features=in_f, drop_rate=0.5)
        x = ops.convert_to_tensor(np.ones((4, 10, in_f), dtype="float32"))
        out_train = call_compute_mlp(model, x, training=True)
        out_infer = call_compute_mlp(model, x, training=False)
        assert out_train.shape == out_infer.shape

    def test_no_drop_at_inference(self):
        in_f = 64
        model = MLP(input_features=in_f, drop_rate=0.5)
        x = ops.convert_to_tensor(np.ones((2, 8, in_f), dtype="float32"))
        out1 = call_compute_mlp(model, x, training=False)
        out2 = call_compute_mlp(model, x, training=False)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-6)


# ─── MLP function ───────────────────────────────────────────────────────────


class TestMLPFunction:
    def test_returns_keras_model(self):
        model = make_mlp(64)
        assert isinstance(model, keras.Model)

    def test_sublayers_accessible(self):
        model = make_mlp(64)
        assert hasattr(model, "fully_connected_layer_1")
        assert hasattr(model, "fully_connected_layer_2")
        assert hasattr(model, "activation")
        assert hasattr(model, "drop_layer")

    def test_fc1_output_dimension_default_hidden(self):
        in_f = 48
        model = make_mlp(in_f)
        x = ops.ones((1, 3, in_f))
        model(x, training=False)
        w = model.fully_connected_layer_1.get_weights()
        assert w[0].shape == (in_f, in_f)

    def test_fc1_output_dimension_custom_hidden(self):
        in_f, hidden = 32, 128
        model = make_mlp(in_f, hidden=hidden)
        x = ops.ones((1, 3, in_f))
        model(x, training=False)
        w = model.fully_connected_layer_1.get_weights()
        assert w[0].shape == (in_f, hidden)

    def test_fc2_output_dimension_default(self):
        in_f = 48
        model = make_mlp(in_f)
        x = ops.ones((1, 3, in_f))
        model(x, training=False)
        w = model.fully_connected_layer_2.get_weights()
        assert w[0].shape[1] == in_f

    def test_fc2_output_dimension_custom(self):
        in_f, out_f = 64, 16
        model = make_mlp(in_f, out=out_f)
        x = ops.ones((1, 3, in_f))
        model(x, training=False)
        w = model.fully_connected_layer_2.get_weights()
        assert w[0].shape[1] == out_f

    def test_default_activation_is_gelu(self):
        model = make_mlp(64)
        assert isinstance(model.activation, Activation)

    def test_custom_activation_layer(self):
        act = keras.layers.Activation("relu")
        model = MLP(input_features=32, activation_layer=act)
        assert model.activation is act

    def test_custom_activation_function(self):
        model = MLP(
            input_features=32,
            activation_layer=keras.activations.gelu,
        )
        assert model.activation is keras.activations.gelu

    def test_use_bias_true_by_default(self):
        model = make_mlp(64)
        x = ops.ones((1, 4, 64))
        model(x, training=False)
        assert model.fully_connected_layer_1.bias is not None
        assert model.fully_connected_layer_2.bias is not None

    def test_use_bias_false(self):
        model = MLP(input_features=64, use_bias=False)
        x = ops.ones((1, 4, 64))
        model(x, training=False)
        assert model.fully_connected_layer_1.bias is None
        assert model.fully_connected_layer_2.bias is None

    def test_forward_pass_output_shape(self):
        model = make_mlp(64)
        x = ops.ones((2, 10, 64))
        out = model(x)
        assert out.shape == (2, 10, 64)

    def test_forward_pass_training_false(self):
        model = make_mlp(64)
        x = ops.convert_to_tensor(np.random.randn(2, 7, 64).astype("float32"))
        out = model(x, training=False)
        assert out.shape == (2, 7, 64)

    def test_forward_pass_training_true(self):
        model = make_mlp(64)
        x = ops.convert_to_tensor(np.random.randn(2, 7, 64).astype("float32"))
        out = model(x, training=True)
        assert out.shape == (2, 7, 64)

    def test_deterministic_inference(self):
        model = make_mlp(64)
        x = ops.convert_to_tensor(np.random.randn(2, 6, 64).astype("float32"))
        out1 = model(x, training=False)
        out2 = model(x, training=False)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-6)

    def test_different_input_sizes(self):
        for in_f in [16, 64, 256]:
            model = make_mlp(in_f)
            x = ops.ones((1, 4, in_f))
            out = model(x, training=False)
            assert out.shape == (1, 4, in_f)

    def test_fc1_set_weights(self):
        model = make_mlp(32)
        x = ops.ones((1, 3, 32))
        model(x, training=False)
        w_shape = model.fully_connected_layer_1.get_weights()[0].shape
        zeros = np.zeros(w_shape, dtype="float32")
        b_shape = model.fully_connected_layer_1.get_weights()[1].shape
        b_zeros = np.zeros(b_shape, dtype="float32")
        model.fully_connected_layer_1.set_weights([zeros, b_zeros])
        got = model.fully_connected_layer_1.get_weights()[0]
        np.testing.assert_array_equal(got, zeros)

    def test_fc2_set_weights(self):
        model = make_mlp(32)
        x = ops.ones((1, 3, 32))
        model(x, training=False)
        w_shape = model.fully_connected_layer_2.get_weights()[0].shape
        zeros = np.zeros(w_shape, dtype="float32")
        b_shape = model.fully_connected_layer_2.get_weights()[1].shape
        b_zeros = np.zeros(b_shape, dtype="float32")
        model.fully_connected_layer_2.set_weights([zeros, b_zeros])
        got = model.fully_connected_layer_2.get_weights()[0]
        np.testing.assert_array_equal(got, zeros)

    def test_fc1_sublayer_callable(self):
        model = make_mlp(64)
        x = ops.ones((1, 4, 64))
        model(x, training=False)
        out = model.fully_connected_layer_1(x)
        assert out.shape == (1, 4, 64)

    def test_activation_sublayer_callable(self):
        model = make_mlp(64)
        x = ops.ones((1, 4, 64))
        model(x, training=False)
        fc1_out = model.fully_connected_layer_1(x)
        act_out = model.activation(fc1_out)
        assert act_out.shape == (1, 4, 64)

    def test_fc2_sublayer_callable(self):
        model = make_mlp(64)
        x = ops.ones((1, 4, 64))
        model(x, training=False)
        fc1_out = model.fully_connected_layer_1(x)
        act_out = model.activation(fc1_out)
        fc2_out = model.fully_connected_layer_2(act_out)
        assert fc2_out.shape == (1, 4, 64)

    def test_manual_subcomponent_chain_matches_model(self):
        model = make_mlp(64)
        x = ops.convert_to_tensor(np.random.randn(2, 8, 64).astype("float32"))
        out_model = model(x, training=False)
        fc1_out = model.fully_connected_layer_1(x)
        act_out = model.activation(fc1_out)
        fc2_out = model.fully_connected_layer_2(act_out)
        np.testing.assert_allclose(np.array(out_model), np.array(fc2_out), atol=1e-5)


# ─── Integration with Block ─────────────────────────────────────────────────


class TestMLPInBlock:
    def test_block_mlp_is_keras_model(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert isinstance(block.mlp, keras.Model)

    def test_block_mlp_has_fc1(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert hasattr(block.mlp, "fully_connected_layer_1")

    def test_block_mlp_has_fc2(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert hasattr(block.mlp, "fully_connected_layer_2")

    def test_block_mlp_has_activation(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert hasattr(block.mlp, "activation")

    def test_block_mlp_fc1_callable(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        norm_out = block.normalization2(x)
        fc1_out = block.mlp.fully_connected_layer_1(norm_out)
        hidden = int(64 * 4.0)
        assert fc1_out.shape == (1, 4, hidden)

    def test_block_mlp_activation_callable(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        norm_out = block.normalization2(x)
        fc1_out = block.mlp.fully_connected_layer_1(norm_out)
        act_out = block.mlp.activation(fc1_out)
        hidden = int(64 * 4.0)
        assert act_out.shape == (1, 4, hidden)

    def test_block_mlp_fc2_callable(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        norm_out = block.normalization2(x)
        fc1_out = block.mlp.fully_connected_layer_1(norm_out)
        act_out = block.mlp.activation(fc1_out)
        fc2_out = block.mlp.fully_connected_layer_2(act_out)
        assert fc2_out.shape == (1, 4, 64)

    def test_block_mlp_weight_porting_pattern(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        fc1_weights = block.mlp.fully_connected_layer_1.get_weights()
        hidden = int(64 * 4.0)
        assert fc1_weights[0].shape == (64, hidden)
        fc2_weights = block.mlp.fully_connected_layer_2.get_weights()
        assert fc2_weights[0].shape == (hidden, 64)

    def test_block_forward_pass_with_mlp(self):
        block = Block(dimension=64, number_of_heads=4)
        x = ops.convert_to_tensor(np.random.randn(2, 8, 64).astype("float32"))
        out = block(x, training=False)
        assert out.shape == (2, 8, 64)

    def test_block_mlp_drop_rate(self):
        block = Block(dimension=64, number_of_heads=4, drop_rate=0.1)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert hasattr(block.mlp, "drop_layer")

    def test_block_custom_activation_function(self):
        block = Block(
            dimension=64,
            number_of_heads=4,
            activation_layer=keras.activations.gelu,
        )
        x = ops.ones((1, 4, 64))
        out = block(x, training=False)
        assert out.shape == (1, 4, 64)


# ─── Integration with NestedTensorBlock ─────────────────────────────────────


class TestMLPInNestedTensorBlock:
    def test_nested_block_mlp_is_keras_model(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert isinstance(block.mlp, keras.Model)

    def test_nested_block_mlp_has_sublayers(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4)
        x = ops.ones((1, 4, 64))
        block(x, training=False)
        assert hasattr(block.mlp, "fully_connected_layer_1")
        assert hasattr(block.mlp, "fully_connected_layer_2")
        assert hasattr(block.mlp, "activation")

    def test_nested_block_apply_ffn_residual(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4)
        x = ops.ones((2, 6, 64))
        block(x, training=False)
        out = block.apply_feedforward_network_residual(x)
        assert out.shape == (2, 6, 64)

    def test_nested_block_forward_pass(self):
        block = NestedTensorBlock(dimension=64, number_of_heads=4)
        x = ops.convert_to_tensor(np.random.randn(2, 8, 64).astype("float32"))
        out = block(x, training=False)
        assert out.shape == (2, 8, 64)

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
        assert len(result) == 2
        assert result[0].shape == (2, 6, 64)
        assert result[1].shape == (3, 6, 64)


# ─── Behavioural consistency ────────────────────────────────────────────────


class TestBehaviouralConsistency:
    def test_model_and_compute_mlp_identical(self):
        model = make_mlp(64)
        x = ops.convert_to_tensor(np.random.randn(2, 10, 64).astype("float32"))
        out_model = model(x, training=False)
        out_fn = call_compute_mlp(model, x, training=False)
        np.testing.assert_allclose(np.array(out_model), np.array(out_fn), atol=1e-6)

    def test_subcomponent_chain_matches_model(self):
        model = make_mlp(64)
        x = ops.convert_to_tensor(np.random.randn(3, 7, 64).astype("float32"))
        out_model = model(x, training=False)
        fc1_out = model.fully_connected_layer_1(x)
        act_out = model.activation(fc1_out)
        fc2_out = model.fully_connected_layer_2(act_out)
        np.testing.assert_allclose(np.array(out_model), np.array(fc2_out), atol=1e-5)

    def test_custom_out_features_shape_preserved(self):
        in_f, hidden, out_f = 64, 256, 32
        model = MLP(
            input_features=in_f,
            hidden_features=hidden,
            output_features=out_f,
        )
        x = ops.ones((2, 5, in_f))
        out = model(x, training=False)
        assert out.shape == (2, 5, out_f)
