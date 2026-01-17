from collections import namedtuple

import keras
from keras import ops


ReversibleEmbedding = namedtuple(
    "ReversibleEmbedding", ["embedding", "logit_soft_cap"]
)


def build_rms_norm(name, epsilon, dtype=None):
    return keras.layers.RMSNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name=name,
    )


def apply_tanh_soft_cap(values, soft_cap):
    if soft_cap is None:
        return values
    values = ops.divide(values, soft_cap)
    values = ops.tanh(values)
    return ops.multiply(values, soft_cap)


def _build_inverse_frequencies(rotary_dim, max_wavelength, scaling_factor):
    indices = ops.arange(0, rotary_dim, 2, dtype="float32")
    dim_scale = ops.cast(rotary_dim, "float32")
    frequency_range = indices / dim_scale
    inverse_frequencies = ops.power(
        ops.cast(max_wavelength, "float32"), -frequency_range
    )
    return inverse_frequencies / ops.cast(scaling_factor, "float32")


def _expand_rotary_embedding(embedding, inputs):
    embedding_rank = len(embedding.shape)
    inputs_rank = len(inputs.shape)
    while embedding_rank < inputs_rank:
        embedding = ops.expand_dims(embedding, axis=2)
        embedding_rank += 1
    return embedding


def _compute_rotary_cos_sin(
    inputs,
    start_index,
    max_wavelength,
    scaling_factor,
):
    sequence_length = ops.shape(inputs)[1]
    rotary_dim = ops.shape(inputs)[-1]
    if rotary_dim % 2 != 0:
        raise ValueError(
            "Rotary dimension must be even, got {}".format(rotary_dim)
        )

    positions = ops.arange(sequence_length, dtype="float32")
    positions = positions + ops.cast(start_index, "float32")
    positions = ops.expand_dims(positions, axis=0)

    inverse_frequencies = _build_inverse_frequencies(
        rotary_dim, max_wavelength, scaling_factor
    )
    frequencies = ops.einsum("bi,j->bij", positions, inverse_frequencies)

    embedding = ops.stack((frequencies, frequencies), axis=-2)
    embedding = ops.reshape(
        embedding,
        (
            1,
            ops.shape(frequencies)[1],
            ops.shape(frequencies)[2] * 2,
        ),
    )
    embedding = _expand_rotary_embedding(embedding, inputs)

    cos_emb = ops.cos(embedding)
    sin_emb = ops.sin(embedding)
    return ops.cast(cos_emb, inputs.dtype), ops.cast(sin_emb, inputs.dtype)


def apply_rotary_embedding(
    inputs,
    start_index,
    max_wavelength,
    scaling_factor,
):
    cos_emb, sin_emb = _compute_rotary_cos_sin(
        inputs,
        start_index,
        max_wavelength,
        scaling_factor,
    )
    first_half, second_half = ops.split(inputs, 2, axis=-1)
    half_rotated = ops.stack((-second_half, first_half), axis=-2)
    half_rotated = ops.reshape(half_rotated, ops.shape(inputs))
    return (inputs * cos_emb) + (half_rotated * sin_emb)


def build_reversible_embedding(
    vocabulary_size,
    hidden_dim,
    logit_soft_cap=None,
    dtype=None,
    name="token_embedding",
):
    embedding_layer = keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=hidden_dim,
        embeddings_initializer=keras.initializers.VarianceScaling(
            scale=1.0,
            mode="fan_in",
            distribution="untruncated_normal",
        ),
        dtype=dtype,
        name=name,
    )
    return ReversibleEmbedding(embedding_layer, logit_soft_cap)


def apply_reversible_embedding(embedding_layers, token_ids):
    return embedding_layers.embedding(token_ids)


def apply_reversible_projection(embedding_layers, hidden_states):
    kernel = embedding_layers.embedding.embeddings
    logits = ops.matmul(hidden_states, ops.transpose(kernel))
    return apply_tanh_soft_cap(logits, embedding_layers.logit_soft_cap)


def compute_causal_mask(
    batch_size,
    input_length,
    output_length,
    cache_index=0,
):
    output_positions = ops.arange(output_length, dtype="float32")
    output_positions = output_positions + ops.cast(cache_index, "float32")
    output_positions = ops.expand_dims(output_positions, axis=1)
    input_positions = ops.arange(input_length, dtype="float32")
    mask = ops.expand_dims(output_positions >= input_positions, axis=0)
    return ops.broadcast_to(mask, (batch_size, output_length, input_length))


def merge_padding_and_attention_mask(
    padding_mask,
    attention_mask,
):
    mask = padding_mask
    if mask is not None:
        mask = ops.cast(ops.expand_dims(mask, axis=1), "int32")
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, "int32")
        if mask is None:
            return attention_mask
        return ops.minimum(mask, attention_mask)
    return mask


def clip_float16(values):
    is_float16 = keras.backend.standardize_dtype(values.dtype) == "float16"
    if is_float16:
        return ops.clip(values, -65504, 65504)
    return values

