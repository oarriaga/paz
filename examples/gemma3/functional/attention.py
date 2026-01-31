import numpy as np
import keras
from keras import ops
from keras.layers import Dropout
from keras.layers import EinsumDense
from keras.layers import RMSNormalization
from keras.layers import Softmax

from examples.gemma3.functional.core import apply_rotary_embedding
from examples.gemma3.functional.core import apply_tanh_soft_cap
from examples.gemma3.functional.core import compute_causal_mask
from examples.gemma3.functional.core import merge_padding_and_attention_mask


def build_attention_layers(
    hidden_dim,
    head_dim,
    num_query_heads,
    num_key_value_heads,
    use_query_key_norm=False,
    layer_norm_epsilon=1e-6,
    dropout=0.0,
    dtype=None,
    name_prefix="attention",
):
    query = EinsumDense(
        "btd,ndh->btnh",
        (None, num_query_heads, head_dim),
        dtype=dtype,
        name=name_prefix + "_query",
    )
    key = EinsumDense(
        "btd,kdh->btkh",
        (None, num_key_value_heads, head_dim),
        dtype=dtype,
        name=name_prefix + "_key",
    )
    value = EinsumDense(
        "btd,kdh->btkh",
        (None, num_key_value_heads, head_dim),
        dtype=dtype,
        name=name_prefix + "_value",
    )
    out = EinsumDense(
        "btnh,nhd->btd",
        (None, hidden_dim),
        dtype=dtype,
        name=name_prefix + "_output",
    )
    query_norm = None
    key_norm = None
    if use_query_key_norm:
        query_norm = RMSNormalization(
            epsilon=layer_norm_epsilon, dtype=dtype, name=name_prefix + "_query_norm"
        )
        key_norm = RMSNormalization(
            epsilon=layer_norm_epsilon, dtype=dtype, name=name_prefix + "_key_norm"
        )
    drop = None
    if dropout:
        drop = Dropout(rate=dropout, dtype=dtype, name=name_prefix + "_dropout")
    softmax = Softmax(dtype="float32", name=name_prefix + "_softmax")
    layers = (query, key, value, out, query_norm, key_norm, drop, softmax)
    return layers


def _unpack_cache(cache):
    if cache is None:
        return None, 0, None
    return cache[0], cache[1], cache[2]


def _query_scale(query_head_dim_normalize, head_dim, hidden_dim, num_query_heads):
    if query_head_dim_normalize:
        return 1 / np.sqrt(head_dim)
    divisor = hidden_dim / num_query_heads
    return 1 / np.sqrt(divisor)


def _apply_rotary_norm(tensor, norm_layer, use_norm, index, wave, scale):
    if use_norm and norm_layer is not None:
        tensor = norm_layer(tensor)
    return apply_rotary_embedding(tensor, index, wave, scale)


def _update_cache(cache_data, key_update, value_update, index, cache_mask):
    start = [0, index, 0, 0]
    if cache_mask is not None:
        mask = ops.expand_dims(cache_mask, axis=-1)
        mask = ops.expand_dims(mask, axis=-1)
        key_shape = ops.shape(key_update)
        value_shape = ops.shape(value_update)
        key_orig = ops.slice(cache_data[:, 0, ...], start, key_shape)
        value_orig = ops.slice(cache_data[:, 1, ...], start, value_shape)
        key_update = ops.where(mask, key_update, key_orig)
        value_update = ops.where(mask, value_update, value_orig)
    key = ops.slice_update(cache_data[:, 0, ...], start, key_update)
    value = ops.slice_update(cache_data[:, 1, ...], start, value_update)
    return ops.stack((key, value), axis=1)


def _reshape_query(query, num_query_heads, num_key_value_heads):
    query_shape = ops.shape(query)
    group = num_query_heads // num_key_value_heads
    shape = query_shape[:-2]
    shape = (*shape, num_key_value_heads, group, query_shape[-1])
    query = ops.reshape(query, shape)
    batch_size, query_length, _, _, head_dim = ops.shape(query)
    return query, batch_size, query_length, head_dim


def _attention_logits(query, key, logit_soft_cap):
    logits = ops.einsum("btkgh,bskh->bkgts", query, key)
    return apply_tanh_soft_cap(logits, logit_soft_cap)


def _attention_weights(logits, softmax, mask, drop, dropout, train):
    softmax_mask = None
    if mask is not None:
        softmax_mask = mask[:, None, None, :, :]
    original_dtype = logits.dtype
    softmax_values = softmax(logits, mask=softmax_mask)
    softmax_values = ops.cast(softmax_values, original_dtype)
    if dropout and drop is not None:
        softmax_values = drop(softmax_values, training=train)
    return softmax_values


def _attention_output(attention_weights, value, batch_size, query_length, num_query_heads, head_dim):
    results = ops.einsum("bkgts,bskh->btkgh", attention_weights, value)
    shape = (batch_size, query_length, num_query_heads, head_dim)
    return ops.reshape(results, shape)


def _mask_no_tokens(attention_vec, mask):
    if mask is None:
        return attention_vec
    no_tokens = ops.all(ops.equal(mask, 0), axis=-1, keepdims=True)
    zeros = ops.zeros_like(attention_vec)
    zero_mask = no_tokens[..., None]
    return ops.where(zero_mask, zeros, attention_vec)


def _compute_bidirectional_sliding_mask(batch_size, length, window_size):
    indices = ops.arange(length)
    right = ops.broadcast_to(indices, (batch_size, length, length))
    left = ops.transpose(right, (0, 2, 1))
    lower = ops.greater_equal(right, left - (window_size - 1))
    upper = ops.less_equal(right, left + (window_size - 1))
    return ops.logical_and(lower, upper)


def _mask_sliding_window(attention_mask, window, bidir, cache_index=0):
    batch_size, query_length, key_length = ops.shape(attention_mask)
    if bidir:
        mask = _compute_bidirectional_sliding_mask(
            batch_size, query_length, window
        )
        return ops.logical_and(attention_mask, mask)
    all_ones = ops.ones((key_length, key_length), "bool")
    if keras.config.backend() == "tensorflow":
        import tensorflow as tf

        band_size = ops.minimum(key_length, window - 1)
        band_size = ops.cast(band_size, "int32")
        sliding_mask = tf.linalg.band_part(all_ones, band_size, band_size)
    else:
        sliding_mask = ops.triu(all_ones, -1 * window + 1)
        sliding_mask = sliding_mask * ops.tril(all_ones, window - 1)
    start = (cache_index, 0)
    sliding_mask = ops.slice(sliding_mask, start, (query_length, key_length))
    sliding_mask = ops.expand_dims(sliding_mask, 0)
    return ops.logical_and(attention_mask, ops.cast(sliding_mask, "bool"))


def _compute_image_bidirectional_attention_mask(vision_mask):
    bidirectional_mask = vision_mask
    pad = [(0, 0), (1, 0)]
    padded = ops.pad(bidirectional_mask, pad, constant_values=0)
    padded = ops.cast(padded, dtype="int32")
    boundary = ops.greater(padded[..., 1:], padded[..., :-1])
    boundary = ops.cast(boundary, dtype="int32")
    numbered = ops.cumsum(boundary, -1)
    indices = ops.multiply(bidirectional_mask, numbered)
    left = ops.expand_dims(indices, 1)
    right = ops.expand_dims(indices, -1)
    return ops.logical_and(ops.equal(left, right), right)


def _compute_attention_mask(inputs, pad_mask, vision_mask, cache, index, bidir):
    decoder_mask = merge_padding_and_attention_mask(pad_mask, None)
    batch_size = ops.shape(inputs)[0]
    input_length = ops.shape(inputs)[1]
    output_length = ops.shape(inputs)[1]
    if cache is not None:
        input_length = ops.shape(cache)[2]
    if bidir:
        mask_1 = decoder_mask
        mask_2 = ops.transpose(mask_1, (0, 2, 1))
        return mask_1 * mask_2
    causal_mask = compute_causal_mask(batch_size, input_length, output_length, index)
    if vision_mask is not None:
        vision_mask = _compute_image_bidirectional_attention_mask(vision_mask)
        causal_mask = ops.logical_or(causal_mask, vision_mask)
    if decoder_mask is not None:
        causal_mask = ops.minimum(decoder_mask, causal_mask)
    return causal_mask


def build_attention(
    hidden_dim,
    head_dim,
    num_query_heads,
    num_key_value_heads,
    use_query_key_norm=False,
    query_head_dim_normalize=True,
    use_sliding_window_attention=False,
    sliding_window_size=4096,
    rope_wavelength=10000.0,
    rope_scaling_factor=1.0,
    logit_soft_cap=None,
    use_bidirectional_attention=False,
    layer_norm_epsilon=1e-6,
    dropout=0.0,
    dtype=None,
    name_prefix="attention",
):
    layers = build_attention_layers(
        hidden_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        use_query_key_norm,
        layer_norm_epsilon,
        dropout,
        dtype,
        name_prefix,
    )
    query, key, value, out, query_norm, key_norm, drop, softmax = layers

    def apply_attention(inputs, mask, cache, train):
        cache_data, cache_index, cache_mask = _unpack_cache(cache)
        wave = rope_wavelength
        scale = rope_scaling_factor
        query_scale = _query_scale(
            query_head_dim_normalize, head_dim, hidden_dim, num_query_heads
        )
        if use_sliding_window_attention and mask is not None:
            mask = _mask_sliding_window(
                mask, sliding_window_size, use_bidirectional_attention, cache_index
            )
        query_values = query(inputs)
        query_values = _apply_rotary_norm(
            query_values, query_norm, use_query_key_norm, cache_index, wave, scale
        )
        if cache_data is None:
            key_values = key(inputs)
            key_values = _apply_rotary_norm(
                key_values, key_norm, use_query_key_norm, cache_index, wave, scale
            )
            value_values = value(inputs)
            new_cache = None
        else:
            key_update = key(inputs)
            key_update = _apply_rotary_norm(
                key_update, key_norm, use_query_key_norm, cache_index, wave, scale
            )
            value_update = value(inputs)
            new_cache = _update_cache(
                cache_data, key_update, value_update, cache_index, cache_mask
            )
            key_values = new_cache[:, 0, ...]
            value_values = new_cache[:, 1, ...]
        query_values = query_values * ops.cast(query_scale, dtype=query_values.dtype)
        query_values, batch_size, query_length, head_dim_size = _reshape_query(
            query_values, num_query_heads, num_key_value_heads
        )
        logits = _attention_logits(query_values, key_values, logit_soft_cap)
        weights = _attention_weights(logits, softmax, mask, drop, dropout, train)
        attention_vec = _attention_output(
            weights, value_values, batch_size, query_length, num_query_heads, head_dim_size
        )
        attention_vec = _mask_no_tokens(attention_vec, mask)
        output = out(attention_vec)
        if cache_data is not None:
            return output, new_cache
        return output

    return apply_attention, layers


build_gemma3_attention = build_attention
