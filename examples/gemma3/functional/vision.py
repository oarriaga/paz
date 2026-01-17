from collections import namedtuple

import keras
from keras import ops

from examples.gemma3.functional.core import build_rms_norm


VisionEncoderConfig = namedtuple(
    "VisionEncoderConfig",
    [
        "image_size",
        "patch_size",
        "num_heads",
        "hidden_dim",
        "num_layers",
        "intermediate_dim",
        "output_dim",
        "pool_size",
        "layer_norm_epsilon",
        "dropout",
    ],
)

VisionEmbeddingLayers = namedtuple(
    "VisionEmbeddingLayers",
    [
        "patch_embedding",
        "position_embedding",
        "position_ids",
        "num_patches",
        "hidden_dim",
    ],
)

VisionAttentionLayers = namedtuple(
    "VisionAttentionLayers",
    [
        "query_proj",
        "key_proj",
        "value_proj",
        "out_proj",
        "dropout_layer",
        "num_heads",
        "hidden_dim",
        "head_dim",
    ],
)

VisionEncoderLayerLayers = namedtuple(
    "VisionEncoderLayerLayers",
    [
        "attention_layers",
        "layer_norm_1",
        "layer_norm_2",
        "mlp_dense_1",
        "mlp_dense_2",
    ],
)

VisionPoolingLayers = namedtuple(
    "VisionPoolingLayers",
    [
        "average_pooling",
        "width",
        "reduced_width",
        "image_size",
        "patch_size",
        "pool_size",
    ],
)

VisionOutputLayers = namedtuple(
    "VisionOutputLayers",
    [
        "soft_embedding_norm",
        "input_projection",
    ],
)

VisionEncoderLayers = namedtuple(
    "VisionEncoderLayers",
    [
        "embedding_layers",
        "encoder_layers",
        "encoder_layer_norm",
        "pooling_layers",
        "output_layers",
    ],
)


def build_vision_encoder_config(
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
):
    return VisionEncoderConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        intermediate_dim=intermediate_dim,
        output_dim=output_dim,
        pool_size=pool_size,
        layer_norm_epsilon=layer_norm_epsilon,
        dropout=dropout,
    )


def compute_num_vision_tokens_per_image(config):
    return ((config.image_size // config.patch_size) ** 2) // (
        config.pool_size**2
    )


def build_vision_embedding_layers(config, dtype=None, name_prefix="vision"):
    patch_embedding = keras.layers.Conv2D(
        filters=config.hidden_dim,
        kernel_size=config.patch_size,
        strides=config.patch_size,
        padding="valid",
        activation=None,
        dtype=dtype,
        name="{}_embedding_conv".format(name_prefix),
    )
    num_patches = (config.image_size // config.patch_size) ** 2
    position_embedding = keras.layers.Embedding(
        input_dim=num_patches,
        output_dim=config.hidden_dim,
        dtype=dtype,
        name="{}_position_embedding".format(name_prefix),
    )
    position_ids = ops.expand_dims(ops.arange(num_patches), axis=0)
    return VisionEmbeddingLayers(
        patch_embedding=patch_embedding,
        position_embedding=position_embedding,
        position_ids=position_ids,
        num_patches=num_patches,
        hidden_dim=config.hidden_dim,
    )


def apply_vision_embedding(layers, inputs):
    hidden = layers.patch_embedding(inputs)
    hidden_shape = ops.shape(hidden)
    hidden = ops.reshape(
        hidden, [hidden_shape[0], layers.num_patches, layers.hidden_dim]
    )
    return hidden + layers.position_embedding(layers.position_ids)


def build_vision_attention_layers(
    config, dtype=None, name_prefix="vision_attention"
):
    head_dim = config.hidden_dim // config.num_heads
    if head_dim * config.num_heads != config.hidden_dim:
        raise ValueError(
            "hidden_dim must be divisible by num_heads (got hidden_dim={} "
            "and num_heads={})".format(config.hidden_dim, config.num_heads)
        )

    dropout_layer = keras.layers.Dropout(
        config.dropout,
        dtype=dtype,
        name="{}_dropout".format(name_prefix),
    )
    query_proj = keras.layers.Dense(
        units=config.hidden_dim,
        dtype=dtype,
        name="{}_query_proj".format(name_prefix),
    )
    key_proj = keras.layers.Dense(
        units=config.hidden_dim,
        dtype=dtype,
        name="{}_key_proj".format(name_prefix),
    )
    value_proj = keras.layers.Dense(
        units=config.hidden_dim,
        dtype=dtype,
        name="{}_value_proj".format(name_prefix),
    )
    out_proj = keras.layers.Dense(
        units=config.hidden_dim,
        dtype=dtype,
        name="{}_out_proj".format(name_prefix),
    )
    return VisionAttentionLayers(
        query_proj=query_proj,
        key_proj=key_proj,
        value_proj=value_proj,
        out_proj=out_proj,
        dropout_layer=dropout_layer,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        head_dim=head_dim,
    )


def _transpose_for_scores(tensor, num_heads, head_dim):
    sequence_length = ops.shape(tensor)[1]
    tensor = ops.reshape(
        tensor, (ops.shape(tensor)[0], sequence_length, num_heads, head_dim)
    )
    return ops.transpose(tensor, axes=[0, 2, 1, 3])


def apply_vision_attention(layers, inputs, attention_mask=None, training=False):
    batch_size = ops.shape(inputs)[0]
    mixed_query = layers.query_proj(inputs=inputs)
    mixed_key = layers.key_proj(inputs=inputs)
    mixed_value = layers.value_proj(inputs=inputs)

    query_layer = _transpose_for_scores(
        mixed_query, layers.num_heads, layers.head_dim
    )
    key_layer = _transpose_for_scores(
        mixed_key, layers.num_heads, layers.head_dim
    )
    value_layer = _transpose_for_scores(
        mixed_value, layers.num_heads, layers.head_dim
    )

    attention_scores = ops.matmul(
        query_layer, ops.transpose(key_layer, axes=[0, 1, 3, 2])
    )
    dk = ops.cast(ops.sqrt(layers.head_dim), dtype=attention_scores.dtype)
    attention_scores = ops.divide(attention_scores, dk)

    if attention_mask is not None:
        attention_scores = ops.add(attention_scores, attention_mask)

    attention_probs = ops.softmax(attention_scores, axis=-1)
    dropout_probs = layers.dropout_layer(
        inputs=attention_probs, training=training
    )

    attention_output = ops.matmul(dropout_probs, value_layer)
    attention_output = ops.transpose(attention_output, axes=[0, 2, 1, 3])

    sequence_length = ops.shape(attention_output)[1]
    attention_output = ops.reshape(
        attention_output, (batch_size, sequence_length, layers.hidden_dim)
    )
    attention_output = layers.out_proj(attention_output, training=training)
    return attention_output, attention_probs


def build_vision_encoder_layer_layers(
    config, dtype=None, name_prefix="vision_block"
):
    attention_layers = build_vision_attention_layers(
        config, dtype=dtype, name_prefix="{}_attention".format(name_prefix)
    )
    layer_norm_1 = keras.layers.LayerNormalization(
        epsilon=config.layer_norm_epsilon,
        dtype=dtype,
        name="{}_layer_norm_1".format(name_prefix),
    )
    mlp_dense_1 = keras.layers.Dense(
        config.intermediate_dim,
        dtype=dtype,
        name="{}_mlp_dense_1".format(name_prefix),
    )
    mlp_dense_2 = keras.layers.Dense(
        config.hidden_dim,
        dtype=dtype,
        name="{}_mlp_dense_2".format(name_prefix),
    )
    layer_norm_2 = keras.layers.LayerNormalization(
        epsilon=config.layer_norm_epsilon,
        dtype=dtype,
        name="{}_layer_norm_2".format(name_prefix),
    )
    return VisionEncoderLayerLayers(
        attention_layers=attention_layers,
        layer_norm_1=layer_norm_1,
        layer_norm_2=layer_norm_2,
        mlp_dense_1=mlp_dense_1,
        mlp_dense_2=mlp_dense_2,
    )


def apply_vision_encoder_layer(layers, inputs, mask=None, training=False):
    residual = inputs
    hidden = layers.layer_norm_1(inputs)
    if mask is not None:
        mask = ops.cast(mask, dtype=hidden.dtype)
    hidden = apply_vision_attention(
        layers.attention_layers,
        hidden,
        attention_mask=mask,
        training=training,
    )[0]
    hidden = hidden + residual

    residual = hidden
    hidden = layers.layer_norm_2(residual)
    hidden = layers.mlp_dense_1(hidden)
    hidden = keras.activations.gelu(hidden, approximate=True)
    hidden = layers.mlp_dense_2(hidden)
    return residual + hidden


def build_vision_pooling_layers(config, dtype=None, name_prefix="vision"):
    width = config.image_size // config.patch_size
    reduced_width = width // config.pool_size
    average_pooling = keras.layers.AveragePooling2D(
        pool_size=config.pool_size,
        strides=config.pool_size,
        padding="valid",
        dtype=dtype,
        name="{}_average_pooling".format(name_prefix),
    )
    return VisionPoolingLayers(
        average_pooling=average_pooling,
        width=width,
        reduced_width=reduced_width,
        image_size=config.image_size,
        patch_size=config.patch_size,
        pool_size=config.pool_size,
    )


def apply_vision_pooling(layers, inputs):
    batch_size, _, hidden_dim = ops.shape(inputs)
    hidden = ops.reshape(inputs, (batch_size, layers.width, layers.width, hidden_dim))
    hidden = layers.average_pooling(hidden)
    return ops.reshape(
        hidden,
        (batch_size, layers.reduced_width * layers.reduced_width, hidden_dim),
    )


def build_vision_output_layers(
    config, dtype=None, name_prefix="vision_output"
):
    soft_embedding_norm = build_rms_norm(
        "{}_soft_embedding_norm".format(name_prefix),
        epsilon=config.layer_norm_epsilon,
        dtype=dtype,
    )
    input_projection = keras.layers.Dense(
        units=config.output_dim,
        use_bias=False,
        kernel_initializer=keras.initializers.RandomNormal(
            mean=0.0, stddev=0.01
        ),
        dtype=dtype,
        name="{}_input_projection".format(name_prefix),
    )
    return VisionOutputLayers(
        soft_embedding_norm=soft_embedding_norm,
        input_projection=input_projection,
    )


def apply_vision_output(layers, inputs):
    hidden = layers.soft_embedding_norm(inputs)
    return layers.input_projection(hidden)


def build_vision_encoder_layers(config, dtype=None, name_prefix="vision"):
    embedding_layers = build_vision_embedding_layers(
        config, dtype=dtype, name_prefix="{}_embedding".format(name_prefix)
    )
    encoder_layers = [
        build_vision_encoder_layer_layers(
            config,
            dtype=dtype,
            name_prefix="{}_block_{}".format(name_prefix, layer_index),
        )
        for layer_index in range(config.num_layers)
    ]
    encoder_layer_norm = keras.layers.LayerNormalization(
        epsilon=config.layer_norm_epsilon,
        dtype=dtype,
        name="{}_encoder_layer_norm".format(name_prefix),
    )
    pooling_layers = build_vision_pooling_layers(
        config, dtype=dtype, name_prefix="{}_pooling".format(name_prefix)
    )
    output_layers = build_vision_output_layers(
        config, dtype=dtype, name_prefix="{}_output".format(name_prefix)
    )
    return VisionEncoderLayers(
        embedding_layers=embedding_layers,
        encoder_layers=encoder_layers,
        encoder_layer_norm=encoder_layer_norm,
        pooling_layers=pooling_layers,
        output_layers=output_layers,
    )


def apply_vision_encoder(
    layers,
    config,
    images,
    mask=None,
    training=False,
):
    inputs_shape = ops.shape(images)
    images = ops.reshape(
        images, [inputs_shape[0] * inputs_shape[1]] + list(inputs_shape[2:])
    )

    hidden = apply_vision_embedding(layers.embedding_layers, images)
    for encoder_layer in layers.encoder_layers:
        hidden = apply_vision_encoder_layer(
            encoder_layer, hidden, mask=mask, training=training
        )
    hidden = layers.encoder_layer_norm(hidden)
    hidden = apply_vision_pooling(layers.pooling_layers, hidden)
    return apply_vision_output(layers.output_layers, hidden)


def build_vision_encoder_model(config, dtype=None, name="vision_encoder"):
    if dtype == "bfloat16":
        dtype = "float32"

    image_input = keras.Input(
        shape=(None, config.image_size, config.image_size, 3),
        name="images",
    )
    layers = build_vision_encoder_layers(config, dtype=dtype, name_prefix=name)
    outputs = apply_vision_encoder(layers, config, image_input)
    model = keras.Model(inputs=image_input, outputs=outputs, name=name)
    return model, layers

