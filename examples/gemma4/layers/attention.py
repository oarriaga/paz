from keras import ops
from keras.layers import Dropout, EinsumDense, Softmax

from .core import apply_partial_rotary_embedding
from .core import apply_tanh_soft_cap
from .core import MergeDims, SplitDim
from .normalization import build_rms_norm, build_v_norm


def attend(x, mask, head_dim, num_query_heads, num_kv_heads, epsilon, wavelength, scaling_factor, partial_rotary, soft_cap, dropout, dtype, name):  # fmt: skip
    query = project_query(
        x, num_query_heads, head_dim, epsilon, dtype, name)
    key = project_key(
        x, num_kv_heads, head_dim, epsilon, dtype, name)
    value = project_value(
        x, num_kv_heads, head_dim, epsilon, dtype, name)
    rope_args = (wavelength, scaling_factor, partial_rotary)
    query = apply_partial_rotary_embedding(query, *rope_args)
    key = apply_partial_rotary_embedding(key, *rope_args)
    attn_args = (query, key, value, mask, num_query_heads,
                 num_kv_heads, head_dim)
    output = compute_attention(
        *attn_args, soft_cap, dropout, dtype, name)
    output = zero_masked_positions(output, mask)
    return project_output(output, x.shape[-1], dtype, name)


def cached_attend(x, cache, cache_index, head_dim, num_query_heads, num_kv_heads, epsilon, wavelength, scaling_factor, partial_rotary, soft_cap, dtype, name, cache_head_dim=None, shared_kv_cache=None):  # fmt: skip
    """Single-step cached attention.

    When shared_kv_cache is provided (kv_shared layers), K/V are read from
    that cache instead of being projected from x.  The layer's own cache slot
    is returned unchanged, matching the reference HF / keras-hub behaviour.
    """
    cache_head_dim = cache_head_dim or head_dim
    query = project_query(
        x, num_query_heads, head_dim, epsilon, dtype, name)
    positions = build_cache_positions(cache_index)
    rope_args = (wavelength, scaling_factor, partial_rotary, positions)
    query = apply_partial_rotary_embedding(query, *rope_args)
    if shared_kv_cache is not None:
        # KV-shared layer: attend using source layer's K/V cache.
        # Own cache slot is left unchanged (no K/V projection issued here).
        kv_src = shared_kv_cache
        updated_cache = cache
    else:
        key = project_key(
            x, num_kv_heads, head_dim, epsilon, dtype, name)
        value = project_value(
            x, num_kv_heads, head_dim, epsilon, dtype, name)
        key = apply_partial_rotary_embedding(key, *rope_args)
        # Ensure K/V match the declared dtype before writing into the cache.
        # Keras may compute in float32 even when dtype="bfloat16".
        key = ops.cast(key, dtype)
        value = ops.cast(value, dtype)
        if head_dim < cache_head_dim:
            key = pad_to_cache_dim(key, cache_head_dim - head_dim)
            value = pad_to_cache_dim(value, cache_head_dim - head_dim)
        updated_cache = update_kv_cache(cache, cache_index, key, value, dtype)
        kv_src = updated_cache
    full_key, full_value = ops.split(kv_src, 2, axis=1)
    full_key = ops.squeeze(full_key, axis=1)
    full_value = ops.squeeze(full_value, axis=1)
    if head_dim < cache_head_dim:
        full_key = full_key[..., :head_dim]
        full_value = full_value[..., :head_dim]
    mask = build_cache_mask(full_key, cache_index)
    attn_args = (query, full_key, full_value, mask,
                 num_query_heads, num_kv_heads, head_dim)
    output = compute_attention(
        *attn_args, soft_cap, 0.0, dtype, name)
    return project_output(output, x.shape[-1], dtype, name), updated_cache


def pad_to_cache_dim(tensor, pad_size):
    ndim = len(tensor.shape)
    padding = [(0, 0)] * (ndim - 1) + [(0, pad_size)]
    return ops.pad(tensor, padding)


def build_cache_mask(full_key, cache_index):
    ones = ops.ones_like(full_key[:, :, 0, 0], dtype="int32")
    positions = ops.cumsum(ones, axis=1) - 1
    threshold = ops.reshape(cache_index, (1, 1))
    mask = ops.cast(ops.less_equal(positions, threshold), "bool")
    return ops.expand_dims(mask, axis=1)


def build_cache_positions(cache_index):
    position = ops.cast(cache_index, "float32")
    return ops.reshape(position, (1, 1))


def update_kv_cache(cache, index, key_update, value_update, dtype=None):
    key_cache = ops.cast(cache[:, 0, ...], dtype) if dtype else cache[:, 0, ...]
    value_cache = ops.cast(cache[:, 1, ...], dtype) if dtype else cache[:, 1, ...]
    start = [0, index, 0, 0]
    key_cache = ops.slice_update(key_cache, start, key_update)
    value_cache = ops.slice_update(value_cache, start, value_update)
    return ops.stack((key_cache, value_cache), axis=1)


def project_query(x, num_heads, head_dim, epsilon, dtype, name):
    shape = (None, num_heads, head_dim)
    equation = "btd,ndh->btnh"
    proj_name = "{}_query".format(name)
    proj = EinsumDense(equation, shape, dtype=dtype, name=proj_name)
    norm = build_rms_norm(epsilon, dtype, "{}_query_norm".format(name))
    return norm(proj(x))


def project_key(x, num_kv_heads, head_dim, epsilon, dtype, name):
    shape = (None, num_kv_heads, head_dim)
    equation = "btd,kdh->btkh"
    proj_name = "{}_key".format(name)
    proj = EinsumDense(equation, shape, dtype=dtype, name=proj_name)
    norm = build_rms_norm(epsilon, dtype, "{}_key_norm".format(name))
    return norm(proj(x))


def project_value(x, num_kv_heads, head_dim, epsilon, dtype, name):
    shape = (None, num_kv_heads, head_dim)
    equation = "btd,kdh->btkh"
    proj_name = "{}_value".format(name)
    proj = EinsumDense(equation, shape, dtype=dtype, name=proj_name)
    norm = build_v_norm(epsilon, dtype, "{}_value_norm".format(name))
    return norm(proj(x))


def compute_attention(query, key, value, mask, num_query_heads, num_kv_heads, head_dim, soft_cap, dropout, dtype, name):  # fmt: skip
    query = reshape_query(query, num_query_heads, num_kv_heads, head_dim)
    logits = ops.einsum("btkgh,bskh->bkgts", query, key)
    logits = apply_tanh_soft_cap(logits, soft_cap)
    if mask is not None:
        mask = mask[:, None, None, :, :]
    softmax_name = "{}_softmax".format(name)
    softmax = Softmax(dtype="float32", name=softmax_name)
    weights = ops.cast(softmax(logits, mask=mask), logits.dtype)
    if dropout:
        drop_name = "{}_dropout".format(name)
        weights = Dropout(dropout, dtype=dtype, name=drop_name)(weights)
    output = ops.einsum("bkgts,bskh->btkgh", weights, value)
    return MergeDims(axis=-3)(output)


def project_output(output, output_dim, dtype, name):
    shape = (None, output_dim)
    proj_name = "{}_attention_output".format(name)
    proj = EinsumDense("btnh,nhd->btd", shape, dtype=dtype, name=proj_name)
    return proj(output)


def zero_masked_positions(output, mask):
    if mask is None:
        return output
    no_tokens = ops.all(
        ops.equal(mask, 0), axis=-1, keepdims=True)
    zeros = ops.zeros_like(output)
    return ops.where(no_tokens[..., None], zeros, output)


def reshape_query(query, num_query_heads, num_kv_heads, head_dim):
    group_size = num_query_heads // num_kv_heads
    sizes = (num_kv_heads, group_size)
    return SplitDim(axis=-2, sizes=sizes)(query)
