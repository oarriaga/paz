import sys
from pathlib import Path

import numpy as np
import keras
from keras import ops

from examples.gemma3.functional.attention import (
    apply_gemma3_attention,
    build_attention_config,
    build_attention_layers,
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

from keras_hub.src.models.gemma3.gemma3_attention import CachedGemma3Attention


def _copy_rms_norm_weights(clean_layer, hub_layer):
    hub_weights = hub_layer.get_weights()
    clean_layer.set_weights([hub_weights[0] + 1.0])


def _copy_attention_weights(clean_layers, hub_layer):
    clean_layers.query_dense.set_weights(hub_layer.query_dense.get_weights())
    clean_layers.key_dense.set_weights(hub_layer.key_dense.get_weights())
    clean_layers.value_dense.set_weights(hub_layer.value_dense.get_weights())
    clean_layers.output_dense.set_weights(hub_layer.output_dense.get_weights())
    if hub_layer.use_query_key_norm:
        _copy_rms_norm_weights(clean_layers.query_norm, hub_layer.query_norm)
        _copy_rms_norm_weights(clean_layers.key_norm, hub_layer.key_norm)


def _build_configs():
    config = build_attention_config(
        hidden_dim=16,
        head_dim=4,
        num_query_heads=4,
        num_key_value_heads=2,
        use_query_key_norm=True,
        query_head_dim_normalize=True,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        rope_wavelength=10_000.0,
        rope_scaling_factor=1.0,
        logit_soft_cap=None,
        dropout=0.0,
    )
    layers = build_attention_layers(config, dtype="float32", name_prefix="test")
    return config, layers


def test_attention_output_matches_hub():
    rng = np.random.default_rng(5)
    inputs = ops.convert_to_tensor(
        rng.standard_normal((2, 3, 16)).astype("float32")
    )
    attention_mask = ops.ones((2, 3, 3), dtype="bool")

    hub_layer = CachedGemma3Attention(
        head_dim=4,
        num_query_heads=4,
        num_key_value_heads=2,
        use_query_key_norm=True,
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        query_head_dim_normalize=True,
        rope_wavelength=10_000.0,
        rope_scaling_factor=1.0,
        dropout=0.0,
        name="hub",
    )
    hub_layer(inputs, attention_mask=attention_mask)

    config, layers = _build_configs()
    apply_gemma3_attention(
        layers,
        config,
        inputs,
        attention_mask=attention_mask,
        training=False,
    )

    _copy_attention_weights(layers, hub_layer)

    hub_output = hub_layer(inputs, attention_mask=attention_mask)
    clean_output = apply_gemma3_attention(
        layers,
        config,
        inputs,
        attention_mask=attention_mask,
        training=False,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_output),
        ops.convert_to_numpy(hub_output),
        rtol=1e-5,
        atol=1e-5,
    )


def test_attention_cache_matches_hub():
    rng = np.random.default_rng(7)
    inputs = ops.convert_to_tensor(
        rng.standard_normal((2, 1, 16)).astype("float32")
    )
    attention_mask = ops.ones((2, 1, 4), dtype="bool")
    cache = ops.zeros((2, 2, 4, 2, 4), dtype="float32")
    cache_update_mask = ops.ones((2, 1), dtype="bool")

    hub_layer = CachedGemma3Attention(
        head_dim=4,
        num_query_heads=4,
        num_key_value_heads=2,
        use_query_key_norm=True,
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        query_head_dim_normalize=True,
        rope_wavelength=10_000.0,
        rope_scaling_factor=1.0,
        dropout=0.0,
        name="hub_cache",
    )
    hub_layer(
        inputs,
        attention_mask=attention_mask,
        cache=cache,
        cache_update_index=2,
        cache_update_mask=cache_update_mask,
    )

    config, layers = _build_configs()
    apply_gemma3_attention(
        layers,
        config,
        inputs,
        attention_mask=attention_mask,
        cache=cache,
        cache_update_index=2,
        cache_update_mask=cache_update_mask,
        training=False,
    )

    _copy_attention_weights(layers, hub_layer)

    _, hub_cache = hub_layer(
        inputs,
        attention_mask=attention_mask,
        cache=cache,
        cache_update_index=2,
        cache_update_mask=cache_update_mask,
    )
    _, clean_cache = apply_gemma3_attention(
        layers,
        config,
        inputs,
        attention_mask=attention_mask,
        cache=cache,
        cache_update_index=2,
        cache_update_mask=cache_update_mask,
        training=False,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_cache),
        ops.convert_to_numpy(hub_cache),
        rtol=1e-5,
        atol=1e-5,
    )
