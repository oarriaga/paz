import sys
from pathlib import Path

import numpy as np
import keras
from keras import ops

from examples.gemma3.functional.core import (
    apply_reversible_embedding,
    apply_reversible_projection,
    apply_rotary_embedding,
    apply_tanh_soft_cap,
    build_reversible_embedding,
    build_rms_norm,
    compute_causal_mask,
    merge_padding_and_attention_mask,
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

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.gemma3.rms_normalization import RMSNormalization


def test_apply_tanh_soft_cap_matches_formula():
    values = ops.convert_to_tensor([[-2.0, 0.5, 3.0]], dtype="float32")
    soft_cap = 2.0
    output = apply_tanh_soft_cap(values, soft_cap)
    expected = np.tanh(np.array([[-2.0, 0.5, 3.0]]) / soft_cap) * soft_cap
    np.testing.assert_allclose(
        ops.convert_to_numpy(output), expected, rtol=1e-6, atol=1e-6
    )


def test_compute_causal_mask_small():
    mask = compute_causal_mask(
        batch_size=1, input_length=4, output_length=2, cache_index=1
    )
    expected = np.array([[[True, True, False, False], [True, True, True, False]]])
    np.testing.assert_array_equal(ops.convert_to_numpy(mask), expected)


def test_merge_padding_and_attention_mask_combines():
    padding_mask = ops.convert_to_tensor([[1, 1, 0]], dtype="int32")
    attention_mask = ops.convert_to_tensor(
        [[[1, 1, 1], [1, 1, 0], [1, 0, 0]]], dtype="int32"
    )
    combined = merge_padding_and_attention_mask(
        padding_mask=padding_mask,
        attention_mask=attention_mask,
    )
    expected = np.array([[[1, 1, 0], [1, 1, 0], [1, 0, 0]]])
    np.testing.assert_array_equal(ops.convert_to_numpy(combined), expected)


def test_apply_rotary_embedding_matches_hub():
    rng = np.random.default_rng(123)
    values = rng.standard_normal((2, 3, 4, 8)).astype("float32")
    inputs = ops.convert_to_tensor(values)
    max_wavelength = 10_000.0
    scaling_factor = 2.0
    start_index = 1

    hub_layer = RotaryEmbedding(
        max_wavelength=max_wavelength, scaling_factor=scaling_factor
    )
    hub_output = hub_layer(inputs, start_index=start_index)
    clean_output = apply_rotary_embedding(
        inputs,
        start_index=start_index,
        max_wavelength=max_wavelength,
        scaling_factor=scaling_factor,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_output),
        ops.convert_to_numpy(hub_output),
        rtol=1e-5,
        atol=1e-5,
    )


def test_rms_norm_matches_hub():
    rng = np.random.default_rng(11)
    values = rng.standard_normal((2, 3, 5)).astype("float32")
    inputs = ops.convert_to_tensor(values)
    epsilon = 1e-6

    hub_layer = RMSNormalization(epsilon=epsilon)
    clean_layer = build_rms_norm("rms_norm", epsilon=epsilon)
    hub_layer(inputs)
    clean_layer(inputs)

    scale_values = rng.standard_normal((5,)).astype("float32")
    hub_layer.set_weights([scale_values])
    clean_layer.set_weights([scale_values + 1.0])

    hub_output = hub_layer(inputs)
    clean_output = clean_layer(inputs)
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_output),
        ops.convert_to_numpy(hub_output),
        rtol=1e-5,
        atol=1e-5,
    )


def test_reversible_projection_uses_embedding_weights():
    embedding_layers = build_reversible_embedding(7, 4, logit_soft_cap=None)
    token_ids = ops.convert_to_tensor([[1, 2, 3]], dtype="int32")
    embeddings = apply_reversible_embedding(embedding_layers, token_ids)
    logits = apply_reversible_projection(embedding_layers, embeddings)
    expected = ops.matmul(
        embeddings, ops.transpose(embedding_layers.embedding.embeddings)
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(logits),
        ops.convert_to_numpy(expected),
        rtol=1e-6,
        atol=1e-6,
    )
