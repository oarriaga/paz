"""Pure attention computation for Gemma4 decoder layers.

These helpers receive already-built sublayers (the projection EinsumDenses and
their norms) and run the math only: RoPE, GQA reshape, softmax, KV-cache update,
masking and soft-cap. The layer that owns the sublayers (Gemma4DecoderLayer)
threads them in; nothing here creates weight-bearing layers.
"""
from keras import ops
from keras.layers import Dropout, Softmax

from paz.layers import MergeDims, SplitDim
from paz.models.transformers import cache as kv_cache
from paz.models.transformers import mask as attention_mask
from paz.models.transformers.embeddings import rotary
from paz.models.transformers.logits import soft_cap as apply_soft_cap


def project(x, projection, norm):
    return norm(projection(x))


def apply_rope(x, wavelength, scaling_factor, partial_rotary, positions=None):
    return rotary.apply_partial(
        x, wavelength, scaling_factor, partial_rotary, positions)


def compute_attention(query, key, value, mask, num_query_heads, num_kv_heads,
                      head_dim, soft_cap, dropout, dtype, name):
    query = reshape_query(query, num_query_heads, num_kv_heads, head_dim)
    logits = ops.einsum("btkgh,bskh->bkgts", query, key)
    logits = apply_soft_cap(logits, soft_cap)
    if mask is not None:
        mask = mask[:, None, None, :, :]
    softmax = Softmax(dtype="float32", name="{}_softmax".format(name))
    weights = ops.cast(softmax(logits, mask=mask), logits.dtype)
    drop = build_dropout(dropout, dtype, name)
    if drop is not None:
        weights = drop(weights)
    output = ops.einsum("bkgts,bskh->btkgh", weights, value)
    return MergeDims(axis=-3)(output)


def build_dropout(rate, dtype, name):
    if not rate:
        return None
    return Dropout(rate, dtype=dtype, name="{}_dropout".format(name))


def reshape_query(query, num_query_heads, num_kv_heads, head_dim):
    group_size = num_query_heads // num_kv_heads
    return SplitDim(axis=-2, sizes=(num_kv_heads, group_size))(query)


def zero_masked_positions(output, mask):
    if mask is None:
        return output
    no_tokens = ops.all(ops.equal(mask, 0), axis=-1, keepdims=True)
    zeros = ops.zeros_like(output)
    return ops.where(no_tokens[..., None], zeros, output)


def update_kv_cache(cache, index, key, value, head_dim, cache_head_dim):
    key = ops.cast(key, cache.dtype)
    value = ops.cast(value, cache.dtype)
    if head_dim < cache_head_dim:
        key = pad_to_cache_dim(key, cache_head_dim - head_dim)
        value = pad_to_cache_dim(value, cache_head_dim - head_dim)
    return kv_cache.update(cache, index, key, value)


def read_kv_cache(kv_source, head_dim, cache_head_dim):
    key, value = ops.split(kv_source, 2, axis=1)
    key = ops.squeeze(key, axis=1)
    value = ops.squeeze(value, axis=1)
    if head_dim < cache_head_dim:
        key = key[..., :head_dim]
        value = value[..., :head_dim]
    return key, value


def pad_to_cache_dim(tensor, pad_size):
    ndim = len(tensor.shape)
    padding = [(0, 0)] * (ndim - 1) + [(0, pad_size)]
    return ops.pad(tensor, padding)


def build_cache_mask(full_key, index, positions, window):
    # Causal (+ optional sliding-window) mask between the query positions and
    # the cache positions. Works for one query (decode) or many (prefill);
    # cumsum avoids arange so the cache length and query count stay dynamic.
    ones = ops.ones_like(full_key[:, :, 0, 0], dtype="int32")
    key_pos = ops.cumsum(ones, axis=1) - 1
    query_pos = query_positions(index, positions)
    mask = attention_mask.causal(query_pos, key_pos)
    if window is not None:
        recent = attention_mask.sliding_window(query_pos, key_pos, window)
        mask = ops.logical_and(mask, recent)
    return ops.cast(mask, "bool")


def query_positions(index, positions):
    if positions is None:
        return ops.reshape(index, (1, 1))
    return ops.reshape(positions, (1, -1))


def build_cache_positions(index, positions):
    transposed = ops.transpose(query_positions(index, positions))
    return ops.cast(transposed, "float32")
