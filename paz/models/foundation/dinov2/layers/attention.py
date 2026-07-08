import keras
from keras import ops
from keras.layers import Dense, Dropout


def attend(x, num_heads, head_dim, attention_dropout_rate,
           projection_dropout_rate, use_qkv_bias, use_projection_bias, name,
           attention_bias=None):  # fmt: skip
    dim = num_heads * head_dim
    scale = head_dim**-0.5
    query_key_value = project_query_key_value(x, dim, use_qkv_bias, name)
    query, key, value = split_query_key_value(query_key_value, num_heads, head_dim)  # fmt: skip
    scores = compute_scores(query, key, scale)
    if attention_bias is not None:
        scores = scores + attention_bias
    context = apply_attention(scores, value, attention_dropout_rate, name)
    context = ops.transpose(context, (0, 2, 1, 3))
    heads = flatten_heads(context)
    rates = (use_projection_bias, projection_dropout_rate)
    return project_output(heads, dim, *rates, name)


def project_query_key_value(x, dim, use_qkv_bias, name):
    return build_attention_dense(dim * 3, use_qkv_bias, f"{name}_qkv")(x)


def project_output(x, dim, use_projection_bias, projection_dropout_rate, name):
    projected = build_attention_dense(dim, use_projection_bias, f"{name}_proj")(x)  # fmt: skip
    return Dropout(projection_dropout_rate, name=f"{name}_proj_drop")(projected)


def build_attention_dense(units, use_bias, name):
    initializer = keras.initializers.TruncatedNormal(stddev=0.02)
    kwargs = dict(use_bias=use_bias, kernel_initializer=initializer, name=name)
    return Dense(units, **kwargs)


def split_query_key_value(query_key_value, num_heads, head_dim):
    # split fused projection into (batch, seq, 3, num_heads, head_dim)
    reshaped = keras.layers.Reshape((-1, 3, num_heads, head_dim))(query_key_value)  # fmt: skip
    heads = ops.transpose(reshaped, (2, 0, 3, 1, 4))
    return heads[0], heads[1], heads[2]


def compute_scores(query, key, scale):
    key_transposed = ops.transpose(key, (0, 1, 3, 2))
    scores = ops.matmul(query, key_transposed)
    return scores * scale


def apply_attention(scores, values, attention_dropout_rate, name):
    probabilities = ops.softmax(scores, axis=-1)
    drop = Dropout(attention_dropout_rate, name=f"{name}_attn_drop")
    return ops.matmul(drop(probabilities), values)


def flatten_heads(tensor):
    num_heads, head_dim = tensor.shape[2], tensor.shape[3]
    return keras.layers.Reshape((-1, num_heads * head_dim))(tensor)
