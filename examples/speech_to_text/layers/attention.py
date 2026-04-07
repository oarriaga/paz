import string

from keras import ops
from keras.layers import Dropout, EinsumDense

from examples.speech_to_text.layers.embedding import Kernel


def index_to_einsum_variable(index):
    return string.ascii_lowercase[index]


def build_projection_equation(free, bound, output):
    input_str, kernel_str, output_str = "", "", ""
    bias_axes, offset = "", 0
    for i in range(free):
        c = index_to_einsum_variable(i + offset)
        input_str += c
        output_str += c
    offset += free
    for i in range(bound):
        c = index_to_einsum_variable(i + offset)
        input_str += c
        kernel_str += c
    offset += bound
    for i in range(output):
        c = index_to_einsum_variable(i + offset)
        kernel_str += c
        output_str += c
        bias_axes += c
    equation = f"{input_str},{kernel_str}->{output_str}"
    return equation, bias_axes, len(output_str)


def build_output_shape(rank, last_dims):
    return [None] * (rank - len(last_dims)) + list(last_dims)


def build_projection(equation, shape, bias_axes, init, name):
    kwargs = {"bias_axes": bias_axes,
              "kernel_initializer": init,
              "bias_initializer": "zeros",
              "name": name}
    return EinsumDense(equation, shape, **kwargs)


def project(x, free, bound, out_dims, shape, use_bias, name):
    args = (free, bound, out_dims)
    equation, bias, rank = build_projection_equation(*args)
    final_shape = build_output_shape(rank - 1, shape)
    bias = bias if use_bias else None
    layer = build_projection(equation, final_shape, bias, Kernel(), name)
    return layer(x)


def expand_mask_for_heads(mask):
    if mask is None:
        return None
    mask = ops.cast(mask, "bool")
    if len(mask.shape) == 2:
        mask = ops.expand_dims(mask, axis=1)
        mask = ops.expand_dims(mask, axis=1)
    elif len(mask.shape) == 3:
        mask = ops.expand_dims(mask, axis=1)
    return mask


def merge_heads(tensor):
    return ops.transpose(tensor, (0, 2, 1, 3))


def mask_scores(scores, mask):
    if mask is None:
        return scores
    neg = ops.cast(-1e9, scores.dtype)
    return ops.where(mask, scores, neg)


def compute_scores(query, key, key_dim):
    scale = ops.sqrt(ops.cast(key_dim, query.dtype))
    query = query / scale
    key = ops.transpose(key, (0, 1, 3, 2))
    return ops.matmul(query, key)


def transpose_to_heads(projection):
    return ops.transpose(projection, (0, 2, 1, 3))


def apply_attention(scores, values, dropout, name):
    probs = ops.softmax(scores, axis=-1)
    drop_name = f"{name}_attention_scores_dropout"
    probs = Dropout(dropout, name=drop_name)(probs)
    return ops.matmul(probs, values)


def project_query(query, num_heads, key_dim, name):
    rank = len(query.shape)
    shape = [num_heads, key_dim]
    return project(query, rank - 1, 1, 2, shape, True, f"{name}_query")


def project_key(key, num_heads, key_dim, name):
    rank = len(key.shape)
    shape = [num_heads, key_dim]
    return project(key, rank - 1, 1, 2, shape, False, f"{name}_key")


def project_value(value, num_heads, key_dim, name):
    rank = len(value.shape)
    shape = [num_heads, key_dim]
    return project(value, rank - 1, 1, 2, shape, True, f"{name}_value")


def project_output(values, query_rank, output_dim, name):
    shape = [output_dim]
    out_name = f"{name}_attention_output"
    return project(values, query_rank - 1, 2, 1, shape, True, out_name)


def attend(query, value, num_heads, key_dim, dropout, name):
    output_dim = query.shape[-1]
    q = transpose_to_heads(project_query(query, num_heads, key_dim, name))
    k = transpose_to_heads(project_key(value, num_heads, key_dim, name))
    v = transpose_to_heads(project_value(value, num_heads, key_dim, name))
    scores = compute_scores(q, k, key_dim)
    output = apply_attention(scores, v, dropout, name)
    output = merge_heads(output)
    query_rank = len(query.shape)
    return project_output(output, query_rank, output_dim, name)


def build_cache(value, key, num_heads, key_dim, name):
    if key is None:
        key = value
    k = project_key(key, num_heads, key_dim, name)
    v = project_value(value, num_heads, key_dim, name)
    return ops.stack((k, v), axis=1)


def update_cache(cache, index, value, key, num_heads, key_dim, name):
    update = build_cache(value, key, num_heads, key_dim, name)
    start = [0, 0, index, 0, 0]
    return ops.slice_update(cache, start, update)


def kv_attend(query, cache, index, value, mask, config, name):
    num_heads = config["num_heads"]
    key_dim = config["hidden_dim"] // num_heads
    dropout = config["dropout"]
    output_dim = query.shape[-1]
    q = project_query(query, num_heads, key_dim, name)
    if index is not None:
        if value is None:
            raise ValueError("Cached updates require value.")
        args = (cache, index, value, None, num_heads, key_dim, name)
        cache = update_cache(*args)
    k_proj, v_proj = cache[:, 0, ...], cache[:, 1, ...]
    q = transpose_to_heads(q)
    k = transpose_to_heads(k_proj)
    v = transpose_to_heads(v_proj)
    scores = compute_scores(q, k, key_dim)
    mask = expand_mask_for_heads(mask)
    scores = mask_scores(scores, mask)
    output = apply_attention(scores, v, dropout, name)
    output = merge_heads(output)
    query_rank = len(query.shape)
    output = project_output(output, query_rank, output_dim, name)
    return output, cache
