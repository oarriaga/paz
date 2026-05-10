import keras
from keras import ops
from keras.layers import Dense, Dropout


def split_query_key_value(qkv, batch_size, num_tokens, num_heads, head_dim):
    shape = (batch_size, num_tokens, 3, num_heads, head_dim)
    qkv = ops.reshape(qkv, shape)
    qkv = ops.transpose(qkv, axes=(2, 0, 3, 1, 4))
    return qkv[0], qkv[1], qkv[2]


def compute_scaled_dot_product_attention(q, k, v, scale, bias, drop, training):
    scores = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2])) * scale
    if bias is not None:
        scores = scores + bias
    scores = ops.softmax(scores, axis=-1)
    scores = drop(scores, training=training)
    return scores @ v


def compute_attention(
    x,
    predict_qkv,
    project,
    attn_drop,
    proj_drop,
    number_of_heads,
    head_dimension,
    scale,
    attention_bias=None,
    training=None,
):
    batch_size, num_tokens, channels = ops.shape(x)
    qkv = predict_qkv(x)
    split_args = qkv, batch_size, num_tokens, number_of_heads, head_dimension
    q, k, v = split_query_key_value(*split_args)
    attn_args = q, k, v, scale, attention_bias, attn_drop, training
    attended = compute_scaled_dot_product_attention(*attn_args)
    output = ops.transpose(attended, axes=(0, 2, 1, 3))
    output = ops.reshape(output, (batch_size, num_tokens, channels))
    return proj_drop(project(output), training=training)


def Attention(
    dimension,
    number_of_heads=8,
    use_query_key_value_bias=False,
    use_projection_bias=True,
    attention_drop_rate=0.0,
    projection_drop_rate=0.0,
    **kwargs,
):
    head_dimension = dimension // number_of_heads
    scale = head_dimension**-0.5
    initializer = keras.initializers.TruncatedNormal(stddev=0.02)
    kw = {"kernel_initializer": initializer}
    predict_query_key_value = Dense(
        dimension * 3, use_bias=use_query_key_value_bias, name="qkv", **kw
    )
    projection_layer = Dense(dimension, use_bias=use_projection_bias, name="proj", **kw)
    attention_drop = Dropout(attention_drop_rate)
    projection_drop = Dropout(projection_drop_rate)

    x_in = keras.Input(shape=(None, dimension))

    def call(x, attention_bias=None, training=None, **_):
        return compute_attention(
            x,
            predict_query_key_value,
            projection_layer,
            attention_drop,
            projection_drop,
            number_of_heads,
            head_dimension,
            scale,
            attention_bias,
            training,
        )

    x_out = compute_attention(
        x_in,
        predict_query_key_value,
        projection_layer,
        attention_drop,
        projection_drop,
        number_of_heads,
        head_dimension,
        scale,
    )
    model = keras.Model(inputs=x_in, outputs=x_out, **kwargs)
    model.predict_query_key_value = predict_query_key_value
    model.projection_layer = projection_layer
    model.attention_drop = attention_drop
    model.projection_drop = projection_drop
    model.number_of_heads = number_of_heads
    model.head_dimension = head_dimension
    model.scale = scale
    model.call = call
    return model
