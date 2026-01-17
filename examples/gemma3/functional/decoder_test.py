import sys
from pathlib import Path

import numpy as np
import keras
from keras import ops

from examples.gemma3.functional.decoder import (
    apply_decoder_block,
    build_decoder_block_config,
    build_decoder_block_layers,
)

KERAS_HUB_ROOT = Path(__file__).resolve().parents[1] / "keras-hub"
if not hasattr(keras.layers, "ReversibleEmbedding"):
    class ReversibleEmbedding(keras.layers.Layer):
        def __init__(
            self,
            input_dim,
            output_dim,
            tie_weights=True,
            embeddings_initializer="uniform",
            logit_soft_cap=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.tie_weights = tie_weights
            self.logit_soft_cap = logit_soft_cap
            self.embedding = keras.layers.Embedding(
                input_dim=input_dim,
                output_dim=output_dim,
                embeddings_initializer=embeddings_initializer,
                dtype=self.dtype_policy,
                name="embedding",
            )

        def build(self, input_shape):
            self.embedding.build(input_shape)
            self.built = True

        def call(self, inputs, reverse=False):
            if not reverse:
                return self.embedding(inputs)
            kernel = self.embedding.embeddings
            logits = ops.matmul(inputs, ops.transpose(kernel))
            if self.logit_soft_cap is None:
                return logits
            logits = ops.divide(logits, self.logit_soft_cap)
            logits = ops.multiply(ops.tanh(logits), self.logit_soft_cap)
            return logits

    keras.layers.ReversibleEmbedding = ReversibleEmbedding
sys.path.insert(0, str(KERAS_HUB_ROOT))

from keras_hub.src.models.gemma3.gemma3_decoder_block import Gemma3DecoderBlock


def _copy_rms_norm_weights(clean_layer, hub_layer):
    hub_weights = hub_layer.get_weights()
    clean_layer.set_weights([hub_weights[0] + 1.0])


def _copy_attention_weights(clean_layers, hub_attention):
    clean_layers.query_dense.set_weights(hub_attention.query_dense.get_weights())
    clean_layers.key_dense.set_weights(hub_attention.key_dense.get_weights())
    clean_layers.value_dense.set_weights(hub_attention.value_dense.get_weights())
    clean_layers.output_dense.set_weights(
        hub_attention.output_dense.get_weights()
    )
    if hub_attention.use_query_key_norm:
        _copy_rms_norm_weights(clean_layers.query_norm, hub_attention.query_norm)
        _copy_rms_norm_weights(clean_layers.key_norm, hub_attention.key_norm)


def test_decoder_block_output_matches_hub():
    rng = np.random.default_rng(9)
    inputs = ops.convert_to_tensor(
        rng.standard_normal((2, 3, 8)).astype("float32")
    )
    padding_mask = ops.ones((2, 3), dtype="int32")

    hub_block = Gemma3DecoderBlock(
        hidden_dim=8,
        intermediate_dim=16,
        head_dim=2,
        num_query_heads=4,
        num_key_value_heads=2,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=True,
        use_post_attention_norm=True,
        gate_dim_reduction=1,
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        layer_norm_epsilon=1e-6,
        rope_wavelength=10_000.0,
        rope_scaling_factor=1.0,
        use_bidirectional_attention=False,
        dropout=0.0,
        name="hub_block",
    )
    hub_block(inputs, padding_mask=padding_mask)

    config = build_decoder_block_config(
        hidden_dim=8,
        intermediate_dim=16,
        head_dim=2,
        num_query_heads=4,
        num_key_value_heads=2,
        query_head_dim_normalize=True,
        use_query_key_norm=True,
        use_post_ffw_norm=True,
        use_post_attention_norm=True,
        gate_dim_reduction=1,
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        layer_norm_epsilon=1e-6,
        rope_wavelength=10_000.0,
        rope_scaling_factor=1.0,
        use_bidirectional_attention=False,
        dropout=0.0,
    )
    layers = build_decoder_block_layers(config, dtype="float32", name_prefix="clean")
    apply_decoder_block(
        layers,
        config,
        inputs,
        padding_mask=padding_mask,
        training=False,
    )

    _copy_rms_norm_weights(layers.pre_attention_norm, hub_block.pre_attention_norm)
    if layers.post_attention_norm is not None:
        _copy_rms_norm_weights(
            layers.post_attention_norm, hub_block.post_attention_norm
        )
    _copy_attention_weights(layers.attention_layers, hub_block.attention)
    _copy_rms_norm_weights(layers.pre_ffw_norm, hub_block.pre_ffw_norm)
    if layers.post_ffw_norm is not None:
        _copy_rms_norm_weights(layers.post_ffw_norm, hub_block.post_ffw_norm)
    layers.gating_ffw.set_weights(hub_block.gating_ffw.get_weights())
    layers.gating_ffw_2.set_weights(hub_block.gating_ffw_2.get_weights())
    layers.ffw_linear.set_weights(hub_block.ffw_linear.get_weights())

    hub_output = hub_block(inputs, padding_mask=padding_mask)
    clean_output = apply_decoder_block(
        layers,
        config,
        inputs,
        padding_mask=padding_mask,
        training=False,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_output),
        ops.convert_to_numpy(hub_output),
        rtol=1e-5,
        atol=1e-5,
    )

