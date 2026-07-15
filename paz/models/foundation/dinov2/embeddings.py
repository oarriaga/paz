from keras import ops
from keras.layers import Conv2D, Reshape

from paz.models.transformers.embeddings.token import LearnableTokens


def build(images, patch_size, hidden_size, num_positions, num_registers):
    patches = build_patch_tokens(images, patch_size, hidden_size)
    tokens = prepend_class_token(patches, hidden_size)
    tokens = add_position_embedding(tokens, num_positions, hidden_size)
    if num_registers:
        tokens = insert_register_tokens(tokens, num_registers, hidden_size)
    return tokens


def build_patch_tokens(images, patch_size, hidden_size):
    kwargs = dict(strides=patch_size, padding="valid", name="patch_embed_proj")
    projection = Conv2D(hidden_size, patch_size, **kwargs)
    return Reshape((-1, hidden_size))(projection(images))


def prepend_class_token(patches, hidden_size):
    class_token = LearnableTokens(1, hidden_size, name="cls_token")(patches)
    return ops.concatenate([class_token, patches], axis=1)


def add_position_embedding(tokens, num_positions, hidden_size):
    table = LearnableTokens(num_positions, hidden_size, name="pos_embed")
    return tokens + table(tokens)


def insert_register_tokens(tokens, num_registers, hidden_size):
    name = "register_tokens"
    registers = LearnableTokens(num_registers, hidden_size, name=name)(tokens)
    head, tail = tokens[:, :1], tokens[:, 1:]
    return ops.concatenate([head, registers, tail], axis=1)
