from keras import ops
from keras.layers import Conv2D, Embedding, Lambda, Reshape

from paz.models.transformers.attention import kernel
from paz.models.transformers.embeddings import absolute


def build_dinov2_embeddings(images, patch_size, hidden_size, num_positions,
                            num_register_tokens):
    patches = build_patch_tokens(images, patch_size, hidden_size)
    tokens = prepend_class_token(patches, hidden_size)
    tokens = add_position_embedding(tokens, num_positions)
    if num_register_tokens:
        tokens = insert_register_tokens(tokens, num_register_tokens,
                                        hidden_size)
    return tokens


def build_patch_tokens(images, patch_size, hidden_size):
    projection = Conv2D(hidden_size, patch_size, strides=patch_size,
                        padding="valid", name="patch_embed_proj")
    return Reshape((-1, hidden_size))(projection(images))


def prepend_class_token(patches, hidden_size):
    class_token = build_learnable_tokens(patches, 1, hidden_size, "cls_token")
    return ops.concatenate([class_token, patches], axis=1)


def insert_register_tokens(tokens, num_register_tokens, hidden_size):
    registers = build_learnable_tokens(tokens, num_register_tokens,
                                        hidden_size, "register_tokens")
    return ops.concatenate([tokens[:, :1], registers, tokens[:, 1:]], axis=1)


def add_position_embedding(tokens, num_positions):
    position = absolute.embed_position(tokens, num_positions, True, None,
                                       "pos_embed")
    return tokens + position


def build_learnable_tokens(reference, count, hidden_size, name):
    indices = build_token_indices(count, f"{name}_indices")(reference)
    table = Embedding(count, hidden_size, kernel(), name=name)(indices)
    return ops.zeros_like(reference[:, :count, :]) + table


def build_token_indices(count, name):
    select = lambda reference: ops.arange(count, dtype="int32")
    return Lambda(select, output_shape=(count,), name=name)
