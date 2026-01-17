from keras import ops


def interleave_embeddings(
    image_embeddings,
    text_embeddings,
    vision_indices,
    num_vision_tokens_per_image,
):
    batch_size, sequence_length, embedding_dim = ops.shape(text_embeddings)
    num_images = ops.shape(image_embeddings)[0]

    flat_text_embeddings = ops.reshape(
        text_embeddings, (batch_size * sequence_length, embedding_dim)
    )
    flat_image_embeddings = ops.reshape(
        image_embeddings,
        (
            num_images * num_vision_tokens_per_image,
            embedding_dim,
        ),
    )

    to_add = ops.multiply(ops.arange(batch_size, dtype="int32"), sequence_length)
    to_add = ops.cast(ops.expand_dims(to_add, axis=-1), "int32")
    vision_indices = ops.add(vision_indices, to_add)

    indices_shape = ops.shape(vision_indices)
    flat_vision_indices = ops.reshape(
        vision_indices, (indices_shape[0] * indices_shape[1], 1)
    )
    indices = ops.cast(flat_vision_indices, "int32")

    zeroth_index_text_embeddings = ops.take(
        flat_text_embeddings, indices=ops.squeeze(to_add, axis=-1), axis=0
    )

    reconstructed_embedding = ops.scatter_update(
        inputs=flat_text_embeddings,
        indices=indices,
        updates=flat_image_embeddings,
    )

    reconstructed_embedding = ops.scatter_update(
        inputs=reconstructed_embedding,
        indices=to_add,
        updates=zeroth_index_text_embeddings,
    )

    reconstructed_embedding = ops.reshape(
        reconstructed_embedding, (batch_size, sequence_length, embedding_dim)
    )
    return reconstructed_embedding

