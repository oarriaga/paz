from keras import ops


def interleave_embeddings(image_embed, text_embed, vision_ids, num_tokens):
    batch_size, seq_len, embed_dim = ops.shape(text_embed)
    num_images = ops.shape(image_embed)[0]
    flat_text = ops.reshape(text_embed, (batch_size * seq_len, embed_dim))
    flat_image = ops.reshape(image_embed, (num_images * num_tokens, embed_dim))
    offsets = ops.arange(batch_size, dtype="int32")
    offsets = ops.multiply(offsets, seq_len)
    offsets = ops.cast(ops.expand_dims(offsets, axis=-1), "int32")
    vision_ids = ops.add(vision_ids, offsets)
    indices_shape = ops.shape(vision_ids)
    flat_ids = ops.reshape(vision_ids, (indices_shape[0] * indices_shape[1], 1))
    indices = ops.cast(flat_ids, "int32")
    zeroth = ops.take(flat_text, indices=ops.squeeze(offsets, axis=-1), axis=0)
    rebuilt = ops.scatter_update(flat_text, indices, flat_image)
    rebuilt = ops.scatter_update(rebuilt, offsets, zeroth)
    rebuilt = ops.reshape(rebuilt, (batch_size, seq_len, embed_dim))
    return rebuilt


def compute_num_vision_tokens_per_image(image_size, patch_size, pool_size):
    size = image_size // patch_size
    return (size * size) // (pool_size**2)
