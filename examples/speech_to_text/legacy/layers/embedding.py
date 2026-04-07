import keras
from keras import ops


def build_position_indices(sequence_length, start_index):
    return ops.arange(start_index, start_index + sequence_length, dtype="int32")


def broadcast_position_embeddings(position_embeddings, inputs):
    return ops.ones_like(inputs) * position_embeddings


def position_embedding(
    inputs,
    sequence_length,
    initializer="glorot_uniform",
    start_index=0,
    positions=None,
    trainable=True,
    dtype="float32",
    name="position_embedding",
):
    feature_size = inputs.shape[-1]
    if feature_size is None:
        raise ValueError("Position embedding inputs must have known feature size.")

    embedding = keras.layers.Embedding(
        sequence_length,
        feature_size,
        embeddings_initializer=keras.initializers.get(initializer),
        trainable=trainable,
        dtype=dtype,
        name=name,
    )
    if positions is None:
        positions = keras.layers.Lambda(
            lambda x: build_position_indices(ops.shape(x)[-2], start_index),
            output_shape=lambda shape: (shape[-2],),
            name=f"{name}_indices",
        )(inputs)
        position_embeddings = embedding(positions)
        position_embeddings = ops.expand_dims(position_embeddings, axis=0)
        return broadcast_position_embeddings(position_embeddings, inputs)

    if len(positions.shape) == 1:
        positions = ops.expand_dims(positions, axis=0)
    position_embeddings = embedding(positions)
    return broadcast_position_embeddings(position_embeddings, inputs)


def token_and_position_embedding(
    token_ids,
    vocabulary_size,
    sequence_length,
    embedding_dim,
    embeddings_initializer,
    start_index=0,
    positions=None,
    dtype="float32",
    token_embedding_name="token_embedding",
    position_embedding_name="position_embedding",
):
    token_embedding = keras.layers.ReversibleEmbedding(
        vocabulary_size,
        embedding_dim,
        tie_weights=True,
        embeddings_initializer=keras.initializers.get(embeddings_initializer),
        mask_zero=False,
        dtype=dtype,
        name=token_embedding_name,
    )
    embedded_tokens = token_embedding(token_ids)
    embedded_positions = position_embedding(
        embedded_tokens,
        sequence_length,
        embeddings_initializer,
        start_index,
        positions,
        True,
        dtype,
        position_embedding_name,
    )
    return embedded_tokens + embedded_positions
