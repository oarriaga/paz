import keras
from keras import ops
from keras.layers import Dense, Dropout


def attend(
    x,
    num_heads,
    head_dim,
    attn_drop_rate,
    proj_drop_rate,
    use_qkv_bias,
    use_proj_bias,
    name,
    attention_bias=None,
):
    dim = num_heads * head_dim
    qkv = project_query_key_value(x, dim, use_qkv_bias, name)
    q, k, v = split_query_key_value(qkv, num_heads, head_dim)
    scale = head_dim**-0.5
    scores = compute_scores(q, k, scale)
    biased = scores if attention_bias is None else scores + attention_bias
    attended = apply_attention(biased, v, attn_drop_rate, name)
    merged = merge_heads(attended)
    flat = flatten_heads(merged)
    output = project_output(flat, dim, use_proj_bias, proj_drop_rate, name)
    return output


def project_query_key_value(x, dim, use_bias, name):
    layer = build_dense(dim * 3, use_bias, f"{name}_qkv")
    return layer(x)


def project_output(x, dim, use_bias, drop_rate, name):
    proj_layer = build_dense(dim, use_bias, f"{name}_proj")
    drop_layer = Dropout(drop_rate, name=f"{name}_proj_drop")
    projected = proj_layer(x)
    return drop_layer(projected)


def build_dense(units, use_bias, name):
    initializer = keras.initializers.TruncatedNormal(stddev=0.02)
    args = dict(use_bias=use_bias, kernel_initializer=initializer, name=name)
    return Dense(units, **args)


def split_query_key_value(qkv, num_heads, head_dim):
    new_shape = (-1, 3, num_heads, head_dim)
    reshaped = keras.layers.Reshape(new_shape)(qkv)
    transposed = ops.transpose(reshaped, (2, 0, 3, 1, 4))
    return transposed[0], transposed[1], transposed[2]


def compute_scores(query, key, scale):
    key_t = ops.transpose(key, (0, 1, 3, 2))
    raw = ops.matmul(query, key_t)
    return raw * scale


def apply_attention(scores, values, drop_rate, name):
    probs = ops.softmax(scores, axis=-1)
    drop = Dropout(drop_rate, name=f"{name}_attn_drop")
    dropped = drop(probs)
    return ops.matmul(dropped, values)


def merge_heads(tensor):
    return ops.transpose(tensor, (0, 2, 1, 3))


def flatten_heads(tensor):
    num_heads = tensor.shape[2]
    head_dim = tensor.shape[3]
    return keras.layers.Reshape((-1, num_heads * head_dim))(tensor)
