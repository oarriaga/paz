import keras
from keras import ops
from keras.layers import AveragePooling2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LayerNormalization
from keras.layers import RMSNormalization


def build_vision_encoder(
    image_size,
    patch_size,
    num_heads,
    hidden_dim,
    num_layers,
    intermediate_dim,
    output_dim,
    pool_size=14,
    layer_norm_epsilon=1e-6,
    dropout=0.0,
    dtype=None,
    name_prefix="vision",
):
    embed_name = name_prefix + "_embedding"
    embed_layers = build_vision_embedding_layers(
        image_size, patch_size, hidden_dim, dtype, embed_name
    )

    encoder_layers = []
    for layer_index in range(num_layers):
        layer_name = "{}_block_{}".format(name_prefix, layer_index)
        layer_layers = build_vision_encoder_layer(
            num_heads,
            hidden_dim,
            intermediate_dim,
            layer_norm_epsilon,
            dropout,
            dtype,
            layer_name,
        )
        encoder_layers.append(layer_layers)

    norm_name = name_prefix + "_encoder_layer_norm"
    encoder_norm = LayerNormalization(
        epsilon=layer_norm_epsilon, dtype=dtype, name=norm_name
    )

    pool_name = name_prefix + "_pooling"
    pooling_layers = build_vision_pooling_layers(
        image_size, patch_size, pool_size, dtype, pool_name
    )

    out_name = name_prefix + "_output"
    output_layers = build_vision_output_layers(
        output_dim, layer_norm_epsilon, dtype, out_name
    )
    layers = (embed_layers, encoder_layers, encoder_norm, pooling_layers)
    layers = layers + (output_layers,)

    def apply_encoder(images, mask, train):
        inputs_shape = ops.shape(images)
        batch = inputs_shape[0] * inputs_shape[1]
        images = ops.reshape(images, [batch] + list(inputs_shape[2:]))
        hidden = apply_vision_embedding(embed_layers, images)
        for layer_layers in encoder_layers:
            hidden = apply_vision_encoder_layer(layer_layers, hidden, mask, train)
        hidden = encoder_norm(hidden)
        hidden = apply_vision_pooling(pooling_layers, hidden)
        return apply_vision_output(output_layers, hidden)

    return apply_encoder, layers


def build_vision_embedding_layers(
    image_size, patch_size, hidden_dim, dtype=None, name_prefix="vision"
):
    emb_name = name_prefix + "_conv"
    patch_embedding = Conv2D(
        filters=hidden_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        dtype=dtype,
        name=emb_name,
    )
    num_patches = (image_size // patch_size) ** 2
    pos_name = name_prefix + "_position_embedding"
    position_embedding = Embedding(
        num_patches, hidden_dim, dtype=dtype, name=pos_name
    )
    return (patch_embedding, position_embedding, num_patches, hidden_dim)


def apply_vision_embedding(layers, inputs):
    patch_embedding = layers[0]
    position_embedding = layers[1]
    num_patches = layers[2]
    hidden_dim = layers[3]
    hidden = patch_embedding(inputs)
    hidden_shape = ops.shape(hidden)
    shape = (hidden_shape[0], num_patches, hidden_dim)
    hidden = ops.reshape(hidden, shape)
    position_ids = ops.expand_dims(ops.arange(num_patches), axis=0)
    return hidden + position_embedding(position_ids)


def build_vision_attention_layers(
    num_heads, hidden_dim, dropout=0.0, dtype=None, name_prefix="vision_attention"
):
    head_dim = hidden_dim // num_heads
    if head_dim * num_heads != hidden_dim:
        msg = "hidden_dim must be divisible by num_heads"
        msg = msg + " (got hidden_dim={} and num_heads={})"
        raise ValueError(msg.format(hidden_dim, num_heads))
    dropout_name = name_prefix + "_dropout"
    dropout_layer = Dropout(dropout, dtype=dtype, name=dropout_name)
    query_name = name_prefix + "_query_proj"
    key_name = name_prefix + "_key_proj"
    value_name = name_prefix + "_value_proj"
    out_name = name_prefix + "_out_proj"
    query_proj = Dense(units=hidden_dim, dtype=dtype, name=query_name)
    key_proj = Dense(units=hidden_dim, dtype=dtype, name=key_name)
    value_proj = Dense(units=hidden_dim, dtype=dtype, name=value_name)
    out_proj = Dense(units=hidden_dim, dtype=dtype, name=out_name)
    layers = (query_proj, key_proj, value_proj, out_proj, dropout_layer)
    layers = layers + (num_heads, hidden_dim, head_dim)
    return layers


def _transpose_for_scores(tensor, num_heads, head_dim):
    sequence_length = ops.shape(tensor)[1]
    shape = (ops.shape(tensor)[0], sequence_length, num_heads, head_dim)
    tensor = ops.reshape(tensor, shape)
    return ops.transpose(tensor, axes=[0, 2, 1, 3])


def apply_vision_attention(layers, inputs, attention_mask, train):
    query_proj = layers[0]
    key_proj = layers[1]
    value_proj = layers[2]
    out_proj = layers[3]
    dropout_layer = layers[4]
    num_heads = layers[5]
    hidden_dim = layers[6]
    head_dim = layers[7]
    mixed_query = query_proj(inputs=inputs)
    mixed_key = key_proj(inputs=inputs)
    mixed_value = value_proj(inputs=inputs)
    query_layer = _transpose_for_scores(mixed_query, num_heads, head_dim)
    key_layer = _transpose_for_scores(mixed_key, num_heads, head_dim)
    value_layer = _transpose_for_scores(mixed_value, num_heads, head_dim)
    key_transpose = ops.transpose(key_layer, axes=[0, 1, 3, 2])
    attention_scores = ops.matmul(query_layer, key_transpose)
    dk = ops.cast(ops.sqrt(head_dim), dtype=attention_scores.dtype)
    attention_scores = ops.divide(attention_scores, dk)
    if attention_mask is not None:
        attention_scores = ops.add(attention_scores, attention_mask)
    attention_probs = ops.softmax(attention_scores, axis=-1)
    dropout_probs = dropout_layer(inputs=attention_probs, training=train)
    attention_output = ops.matmul(dropout_probs, value_layer)
    attention_output = ops.transpose(attention_output, axes=[0, 2, 1, 3])
    seq_len = ops.shape(attention_output)[1]
    shape = (ops.shape(attention_output)[0], seq_len, hidden_dim)
    attention_output = ops.reshape(attention_output, shape)
    attention_output = out_proj(attention_output, training=train)
    return attention_output, attention_probs


def build_vision_encoder_layer(
    num_heads,
    hidden_dim,
    intermediate_dim,
    layer_norm_epsilon=1e-6,
    dropout=0.0,
    dtype=None,
    name_prefix="vision_block",
):
    att_name = name_prefix + "_attention"
    attention_layers = build_vision_attention_layers(
        num_heads, hidden_dim, dropout, dtype, att_name
    )
    norm_name_1 = name_prefix + "_layer_norm_1"
    norm_name_2 = name_prefix + "_layer_norm_2"
    dense_1_name = name_prefix + "_mlp_dense_1"
    dense_2_name = name_prefix + "_mlp_dense_2"
    layer_norm_1 = LayerNormalization(
        epsilon=layer_norm_epsilon, dtype=dtype, name=norm_name_1
    )
    layer_norm_2 = LayerNormalization(
        epsilon=layer_norm_epsilon, dtype=dtype, name=norm_name_2
    )
    mlp_dense_1 = Dense(intermediate_dim, dtype=dtype, name=dense_1_name)
    mlp_dense_2 = Dense(hidden_dim, dtype=dtype, name=dense_2_name)
    return (attention_layers, layer_norm_1, layer_norm_2, mlp_dense_1, mlp_dense_2)


def apply_vision_encoder_layer(layer_layers, inputs, mask, train):
    attention_layers = layer_layers[0]
    layer_norm_1 = layer_layers[1]
    layer_norm_2 = layer_layers[2]
    mlp_dense_1 = layer_layers[3]
    mlp_dense_2 = layer_layers[4]
    residual = inputs
    hidden = layer_norm_1(inputs)
    if mask is not None:
        mask = ops.cast(mask, dtype=hidden.dtype)
    attention_output, _ = apply_vision_attention(
        attention_layers, hidden, mask, train
    )
    hidden = attention_output + residual
    residual = hidden
    hidden = layer_norm_2(residual)
    hidden = mlp_dense_1(hidden)
    hidden = keras.activations.gelu(hidden, approximate=True)
    hidden = mlp_dense_2(hidden)
    return residual + hidden


def build_vision_pooling_layers(
    image_size, patch_size, pool_size, dtype=None, name_prefix="vision"
):
    width = image_size // patch_size
    reduced_width = width // pool_size
    pool_name = name_prefix + "_average_pooling"
    average_pooling = AveragePooling2D(
        pool_size=pool_size,
        strides=pool_size,
        padding="valid",
        dtype=dtype,
        name=pool_name,
    )
    return (average_pooling, width, reduced_width)


def apply_vision_pooling(layers, inputs):
    average_pooling = layers[0]
    width = layers[1]
    reduced_width = layers[2]
    batch_size, _, hidden_dim = ops.shape(inputs)
    shape = (batch_size, width, width, hidden_dim)
    hidden = ops.reshape(inputs, shape)
    hidden = average_pooling(hidden)
    size = reduced_width
    shape = (batch_size, size * size, hidden_dim)
    return ops.reshape(hidden, shape)


def build_vision_output_layers(
    output_dim, layer_norm_epsilon=1e-6, dtype=None, name_prefix="vision_output"
):
    norm_name = name_prefix + "_soft_embedding_norm"
    soft_norm = RMSNormalization(
        epsilon=layer_norm_epsilon, dtype=dtype, name=norm_name
    )
    proj_name = name_prefix + "_input_projection"
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    input_projection = Dense(
        units=output_dim,
        use_bias=False,
        kernel_initializer=initializer,
        dtype=dtype,
        name=proj_name,
    )
    return (soft_norm, input_projection)


def apply_vision_output(layers, inputs):
    soft_norm = layers[0]
    input_projection = layers[1]
    hidden = soft_norm(inputs)
    return input_projection(hidden)
