from collections import namedtuple

import keras
import numpy as np
from keras import ops

from examples.gemma3.functional.core import (
    apply_rotary_embedding,
    apply_tanh_soft_cap,
    build_rms_norm,
)


AttentionConfig = namedtuple(
    "AttentionConfig",
    [
        "hidden_dim",
        "head_dim",
        "num_query_heads",
        "num_key_value_heads",
        "logit_soft_cap",
        "use_sliding_window_attention",
        "sliding_window_size",
        "query_head_dim_normalize",
        "use_query_key_norm",
        "layer_norm_epsilon",
        "rope_wavelength",
        "rope_scaling_factor",
        "use_bidirectional_attention",
        "dropout",
    ],
)

AttentionLayers = namedtuple(
    "AttentionLayers",
    [
        "query_dense",
        "key_dense",
        "value_dense",
        "output_dense",
        "query_norm",
        "key_norm",
        "dropout_layer",
        "softmax",
    ],
)


def build_attention_config(
    hidden_dim,
    head_dim,
    num_query_heads,
    num_key_value_heads,
    logit_soft_cap=None,
    use_sliding_window_attention=False,
    sliding_window_size=4096,
    query_head_dim_normalize=True,
    use_query_key_norm=False,
    layer_norm_epsilon=1e-6,
    rope_wavelength=10_000.0,
    rope_scaling_factor=1.0,
    use_bidirectional_attention=False,
    dropout=0.0,
):
    return AttentionConfig(
        hidden_dim=hidden_dim,
        head_dim=head_dim,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        logit_soft_cap=logit_soft_cap,
        use_sliding_window_attention=use_sliding_window_attention,
        sliding_window_size=sliding_window_size,
        query_head_dim_normalize=query_head_dim_normalize,
        use_query_key_norm=use_query_key_norm,
        layer_norm_epsilon=layer_norm_epsilon,
        rope_wavelength=rope_wavelength,
        rope_scaling_factor=rope_scaling_factor,
        use_bidirectional_attention=use_bidirectional_attention,
        dropout=dropout,
    )


def build_attention_layers(config, dtype=None, name_prefix="attention"):
    query_dense = keras.layers.EinsumDense(
        "btd,ndh->btnh",
        output_shape=(None, config.num_query_heads, config.head_dim),
        dtype=dtype,
        name="{}_query".format(name_prefix),
    )
    key_dense = keras.layers.EinsumDense(
        "bsd,kdh->bskh",
        output_shape=(None, config.num_key_value_heads, config.head_dim),
        dtype=dtype,
        name="{}_key".format(name_prefix),
    )
    value_dense = keras.layers.EinsumDense(
        "bsd,kdh->bskh",
        output_shape=(None, config.num_key_value_heads, config.head_dim),
        dtype=dtype,
        name="{}_value".format(name_prefix),
    )
    output_dense = keras.layers.EinsumDense(
        "btnh,nhd->btd",
        output_shape=(None, config.hidden_dim),
        dtype=dtype,
        name="{}_output".format(name_prefix),
    )

    query_norm = None
    key_norm = None
    if config.use_query_key_norm:
        query_norm = build_rms_norm(
            "{}_query_norm".format(name_prefix),
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
        )
        key_norm = build_rms_norm(
            "{}_key_norm".format(name_prefix),
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
        )

    dropout_layer = None
    if config.dropout:
        dropout_layer = keras.layers.Dropout(
            rate=config.dropout,
            dtype=dtype,
            name="{}_dropout".format(name_prefix),
        )

    softmax = keras.layers.Softmax(
        dtype="float32",
        name="{}_softmax".format(name_prefix),
    )

    return AttentionLayers(
        query_dense=query_dense,
        key_dense=key_dense,
        value_dense=value_dense,
        output_dense=output_dense,
        query_norm=query_norm,
        key_norm=key_norm,
        dropout_layer=dropout_layer,
        softmax=softmax,
    )


def _compute_bidirectional_sliding_mask(
    batch_size,
    sequence_length,
    sliding_window_size,
):
    row_indices = ops.expand_dims(
        ops.arange(sequence_length, dtype="int32"), axis=1
    )
    column_indices = ops.arange(sequence_length, dtype="int32")

    window_right = sliding_window_size // 2
    window_left = sliding_window_size - window_right - 1
    distance = row_indices - column_indices
    mask = ops.logical_and(
        distance <= window_left,
        distance >= -window_right,
    )
    mask = ops.expand_dims(mask, axis=0)
    return ops.broadcast_to(mask, (batch_size, sequence_length, sequence_length))


def _mask_sliding_window(
    attention_mask,
    sliding_window_size,
    use_bidirectional_attention,
    cache_update_index=0,
):
    batch_size, query_length, key_length = ops.shape(attention_mask)

    if use_bidirectional_attention:
        bidirectional_sliding_mask = _compute_bidirectional_sliding_mask(
            batch_size=batch_size,
            sequence_length=query_length,
            sliding_window_size=sliding_window_size,
        )
        return ops.logical_and(attention_mask, bidirectional_sliding_mask)

    all_ones = ops.ones((key_length, key_length), "bool")
    if keras.config.backend() == "tensorflow":
        import tensorflow as tf

        band_size = ops.minimum(key_length, sliding_window_size - 1)
        band_size = ops.cast(band_size, "int32")
        sliding_mask = tf.linalg.band_part(all_ones, band_size, band_size)
    else:
        sliding_mask = ops.triu(all_ones, -1 * sliding_window_size + 1)
        sliding_mask = sliding_mask * ops.tril(all_ones, sliding_window_size - 1)

    start = (cache_update_index, 0)
    sliding_mask = ops.slice(sliding_mask, start, (query_length, key_length))
    sliding_mask = ops.expand_dims(sliding_mask, 0)
    return ops.logical_and(attention_mask, ops.cast(sliding_mask, "bool"))


def _compute_attention(
    layers,
    config,
    query,
    key,
    value,
    attention_mask,
    training=False,
    cache_update_index=0,
):
    if config.query_head_dim_normalize:
        query_normalization = 1 / np.sqrt(config.head_dim)
    else:
        query_normalization = 1 / np.sqrt(
            config.hidden_dim / config.num_query_heads
        )

    if config.use_sliding_window_attention and attention_mask is not None:
        attention_mask = _mask_sliding_window(
            attention_mask=attention_mask,
            sliding_window_size=config.sliding_window_size,
            use_bidirectional_attention=config.use_bidirectional_attention,
            cache_update_index=cache_update_index,
        )

    query = query * ops.cast(query_normalization, dtype=query.dtype)
    query_shape = ops.shape(query)
    query = ops.reshape(
        query,
        (
            *query_shape[:-2],
            config.num_key_value_heads,
            config.num_query_heads // config.num_key_value_heads,
            query_shape[-1],
        ),
    )
    batch_size, query_length, _, _, head_dim = ops.shape(query)

    attention_logits = ops.einsum("btkgh,bskh->bkgts", query, key)
    attention_logits = apply_tanh_soft_cap(
        attention_logits, config.logit_soft_cap
    )

    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :, :]

    original_dtype = attention_logits.dtype
    attention_softmax = layers.softmax(attention_logits, mask=attention_mask)
    attention_softmax = ops.cast(attention_softmax, original_dtype)

    if config.dropout and layers.dropout_layer is not None:
        attention_softmax = layers.dropout_layer(
            attention_softmax, training=training
        )

    results = ops.einsum("bkgts,bskh->btkgh", attention_softmax, value)
    return ops.reshape(results, (batch_size, query_length, config.num_query_heads, head_dim))


def apply_gemma3_attention(
    layers,
    config,
    inputs,
    attention_mask=None,
    cache=None,
    cache_update_index=0,
    cache_update_mask=None,
    training=False,
):
    query = layers.query_dense(inputs)
    if config.use_query_key_norm and layers.query_norm is not None:
        query = layers.query_norm(query)
    query = apply_rotary_embedding(
        query,
        start_index=cache_update_index,
        max_wavelength=config.rope_wavelength,
        scaling_factor=config.rope_scaling_factor,
    )

    if cache is not None:
        key_cache = cache[:, 0, ...]
        value_cache = cache[:, 1, ...]
        key_update = layers.key_dense(inputs)
        if config.use_query_key_norm and layers.key_norm is not None:
            key_update = layers.key_norm(key_update)
        key_update = apply_rotary_embedding(
            key_update,
            start_index=cache_update_index,
            max_wavelength=config.rope_wavelength,
            scaling_factor=config.rope_scaling_factor,
        )
        value_update = layers.value_dense(inputs)

        start = [0, cache_update_index, 0, 0]
        if cache_update_mask is not None:
            cache_update_mask = ops.expand_dims(
                ops.expand_dims(cache_update_mask, axis=-1), axis=-1
            )
            key_original = ops.slice(
                key_cache, start, ops.shape(key_update)
            )
            value_original = ops.slice(
                value_cache, start, ops.shape(value_update)
            )
            key_update = ops.where(cache_update_mask, key_update, key_original)
            value_update = ops.where(
                cache_update_mask, value_update, value_original
            )

        key = ops.slice_update(key_cache, start, key_update)
        value = ops.slice_update(value_cache, start, value_update)
        new_cache = ops.stack((key, value), axis=1)
    else:
        key = layers.key_dense(inputs)
        if config.use_query_key_norm and layers.key_norm is not None:
            key = layers.key_norm(key)
        key = apply_rotary_embedding(
            key,
            start_index=cache_update_index,
            max_wavelength=config.rope_wavelength,
            scaling_factor=config.rope_scaling_factor,
        )
        value = layers.value_dense(inputs)
        new_cache = None

    attention_vec = _compute_attention(
        layers,
        config,
        query,
        key,
        value,
        attention_mask,
        training=training,
        cache_update_index=cache_update_index,
    )

    if attention_mask is not None:
        no_attended_tokens = ops.all(
            ops.equal(attention_mask, 0), axis=-1, keepdims=True
        )[..., None]
        attention_vec = ops.where(
            no_attended_tokens, ops.zeros_like(attention_vec), attention_vec
        )

    attention_output = layers.output_dense(attention_vec)
    if cache is not None:
        return attention_output, new_cache
    return attention_output

