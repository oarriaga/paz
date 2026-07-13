import keras
from keras import ops
from keras.layers import Dense, Dropout


def attend(
    x,
    num_heads,
    head_dim,
    attention_dropout_rate,
    projection_dropout_rate,
    use_QKV_bias,
    use_projection_bias,
    name,
    attention_bias=None,
):
    dim = num_heads * head_dim
    scale = head_dim**-0.5
    QKV = project_query_key_value(x, dim, use_QKV_bias, name)
    Q, K, V = split_query_key_value(QKV, num_heads, head_dim)
    scores = compute_scores(Q, K, scale)
    biased_scores = scores if attention_bias is None else scores + attention_bias
    context = apply_attention(biased_scores, V, attention_dropout_rate, name)
    transposed_heads = ops.transpose(context, (0, 2, 1, 3))
    flat_heads = flatten_heads(transposed_heads)
    return project_output(flat_heads, dim, use_projection_bias, projection_dropout_rate, name)  # fmt: skip


def project_query_key_value(x, dim, use_QKV_bias, name):
    return AttentionDense(dim * 3, use_QKV_bias, f"{name}_qkv")(x)


def project_output(x, dim, use_projection_bias, projection_dropout_rate, name):
    projected = AttentionDense(dim, use_projection_bias, f"{name}_proj")(x)
    return Dropout(projection_dropout_rate, name=f"{name}_proj_drop")(projected)


def AttentionDense(units, use_bias, name):
    initializer = keras.initializers.TruncatedNormal(stddev=0.02)
    kwargs = dict(use_bias=use_bias, kernel_initializer=initializer, name=name)
    return Dense(units, **kwargs)


def split_query_key_value(QKV, num_heads, head_dim):
    # split fused projection into Q, K, V per head; Reshape keeps the batch axis, so -1 is seq_len -> (batch, seq_len, 3, num_heads, head_dim)
    new_shape = (-1, 3, num_heads, head_dim)
    QKV = keras.layers.Reshape(new_shape)(QKV)
    QKV = ops.transpose(QKV, (2, 0, 3, 1, 4))
    return QKV[0], QKV[1], QKV[2]


def compute_scores(Q, K, scale):
    key_transposed = ops.transpose(K, (0, 1, 3, 2))
    scores = ops.matmul(Q, key_transposed)
    return scores * scale


def apply_attention(scores, values, attention_dropout_rate, name):
    probabilities = ops.softmax(scores, axis=-1)
    dropped = Dropout(attention_dropout_rate, name=f"{name}_attn_drop")(probabilities)
    return ops.matmul(dropped, values)


def flatten_heads(tensor):
    num_heads = tensor.shape[2]
    head_dim = tensor.shape[3]
    return keras.layers.Reshape((-1, num_heads * head_dim))(tensor)
