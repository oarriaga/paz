from collections import namedtuple

from keras import ops
from keras.layers import Dropout, EinsumDense, Softmax

from .core import apply_partial_rotary_embedding
from .core import apply_tanh_soft_cap
from .core import MergeDims, SplitDim
from .normalization import build_rms_norm, build_v_norm

AttendArgs = namedtuple(
    "AttendArgs",
    "x mask head_dim num_query_heads num_kv_heads epsilon wavelength "
    "scaling_factor partial_rotary soft_cap dropout dtype name",
)
CachedAttendArgs = namedtuple(
    "CachedAttendArgs",
    "x cache index head_dim num_query_heads num_kv_heads epsilon wavelength "
    "scaling_factor partial_rotary soft_cap dtype name cache_head_dim",
)


def attend(args):
    x = args.x
    dtype = args.dtype
    name = args.name
    query = project_query(
        x, args.num_query_heads, args.head_dim, args.epsilon, dtype, name)
    key = project_key(
        x, args.num_kv_heads, args.head_dim, args.epsilon, dtype, name)
    value = project_value(
        x, args.num_kv_heads, args.head_dim, args.epsilon, dtype, name)
    rope_args = (args.wavelength, args.scaling_factor, args.partial_rotary)
    query = apply_partial_rotary_embedding(query, *rope_args)
    key = apply_partial_rotary_embedding(key, *rope_args)
    output = compute_attention(query, key, value, args.mask, args)
    output = zero_masked_positions(output, args.mask)
    return project_output(output, x.shape[-1], dtype, name)


def cached_attend(args, shared_kv_cache=None):
    """Single-step cached attention.

    When shared_kv_cache is provided (kv_shared layers), K/V are read from
    that cache instead of being projected from x.
    """
    x = args.x
    dtype = args.dtype
    name = args.name
    head_dim = args.head_dim
    cache_head_dim = args.cache_head_dim or head_dim
    query = project_query(
        x, args.num_query_heads, head_dim, args.epsilon, dtype, name)
    positions = build_cache_positions(args.index)
    rope_args = (
        args.wavelength,
        args.scaling_factor,
        args.partial_rotary,
        positions,
    )
    query = apply_partial_rotary_embedding(query, *rope_args)
    if shared_kv_cache is not None:
        kv_src = shared_kv_cache
        updated_cache = args.cache
    else:
        key = project_key(
            x, args.num_kv_heads, head_dim, args.epsilon, dtype, name)
        value = project_value(
            x, args.num_kv_heads, head_dim, args.epsilon, dtype, name)
        key = apply_partial_rotary_embedding(key, *rope_args)
        key = ops.cast(key, dtype)
        value = ops.cast(value, dtype)
        if head_dim < cache_head_dim:
            key = pad_to_cache_dim(key, cache_head_dim - head_dim)
            value = pad_to_cache_dim(value, cache_head_dim - head_dim)
        updated_cache = update_kv_cache(
            args.cache, args.index, key, value, dtype)
        kv_src = updated_cache
    full_key, full_value = ops.split(kv_src, 2, axis=1)
    full_key = ops.squeeze(full_key, axis=1)
    full_value = ops.squeeze(full_value, axis=1)
    if head_dim < cache_head_dim:
        full_key = full_key[..., :head_dim]
        full_value = full_value[..., :head_dim]
    mask = build_cache_mask(full_key, args.index)
    output = compute_attention(query, full_key, full_value, mask, args)
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
    key_cache = cache[:, 0, ...]
    value_cache = cache[:, 1, ...]
    if dtype is not None:
        key_cache = ops.cast(key_cache, dtype)
        value_cache = ops.cast(value_cache, dtype)
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


def compute_attention(query, key, value, mask, args):
    query = reshape_query(
        query, args.num_query_heads, args.num_kv_heads, args.head_dim)
    logits = ops.einsum("btkgh,bskh->bkgts", query, key)
    logits = apply_tanh_soft_cap(logits, args.soft_cap)
    if mask is not None:
        mask = mask[:, None, None, :, :]
    softmax_name = "{}_softmax".format(args.name)
    softmax = Softmax(dtype="float32", name=softmax_name)
    weights = ops.cast(softmax(logits, mask=mask), logits.dtype)
    dropout = build_dropout(args)
    if dropout is not None:
        weights = dropout(weights)
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


def build_dropout(args):
    rate = getattr(args, "dropout", 0.0)
    if not rate:
        return None
    name = "{}_dropout".format(args.name)
    return Dropout(rate, dtype=args.dtype, name=name)
