from functools import partial

import numpy as np
import keras
from keras import Input, Model, layers, ops, initializers

from paz.models.foundation.dinov2.layers import build_patch_embedding, block

CLS_TOKEN = "cls_token"
POS_EMBED = "pos_embed"
REGISTER_TOKENS = "register_tokens"
PATCH_EMBED = "patch_embed"
FINAL_NORM = "norm"


def DINOV2VIT(img_size, patch_size, embed_dim, depth, num_heads, num_register_tokens=0, init_values=1e-5, feedforward_layer="mlp", name=None):  # fmt: skip
    pixels = Input((img_size, img_size, 3), name="pixels")
    patches = build_patch_embedding(pixels, patch_size, embed_dim, PATCH_EMBED)
    num_patches = (img_size // patch_size) ** 2
    tokens = build_token_stream(patches, embed_dim, num_patches, num_register_tokens)  # fmt: skip
    features = build_transformer_blocks(tokens, embed_dim, depth, num_heads, init_values, feedforward_layer)  # fmt: skip
    normalized = layers.LayerNormalization(epsilon=1e-6, name=FINAL_NORM)(features)
    cls_output = take_first_token(normalized)
    return Model(pixels, cls_output, name=name)


def take_first_token(features):
    return features[:, 0]


def build_token_stream(patches, embed_dim, num_patches, num_register_tokens):
    init_cls = initializers.RandomNormal(stddev=1e-6)
    init_pos = initializers.TruncatedNormal(stddev=0.02)
    cls = lookup_table_constant(CLS_TOKEN, 1, embed_dim, init_cls, patches)
    tokens = prepend_cls_token(patches, cls)
    pos = lookup_table_constant(POS_EMBED, num_patches + 1, embed_dim, init_pos, tokens)  # fmt: skip
    tokens = tokens + pos
    return insert_register_tokens(tokens, num_register_tokens, embed_dim)


def lookup_table_constant(name, num, embed_dim, initializer, anchor):
    kwargs = dict(embeddings_initializer=initializer, name=name)
    indices = build_indices(num, anchor, name)
    return layers.Embedding(num, embed_dim, **kwargs)(indices)


def build_indices(num, anchor, name):
    kwargs = dict(arguments={"num": num}, output_shape=(num,), name=f"{name}_indices")  # fmt: skip
    return layers.Lambda(arange_indices, **kwargs)(anchor)


@keras.saving.register_keras_serializable(package="paz_dinov2")
def arange_indices(reference, num=0):
    # registered so ported .keras files using it in a Lambda can deserialize
    return ops.arange(num, dtype="int32")


def prepend_cls_token(patches, cls):
    broadcasted = broadcast_single_token(cls, patches)
    return ops.concatenate([broadcasted, patches], axis=1)


def broadcast_single_token(token, reference):
    # zeros_like carries reference's dynamic batch; ops.broadcast_to with a
    # symbolic batch (ops.shape(reference)[0]) fails at graph-build time
    return ops.zeros_like(reference[:, :1, :]) + token


def insert_register_tokens(tokens, num_register_tokens, embed_dim):
    if num_register_tokens == 0:
        return tokens
    initializer = initializers.RandomNormal(stddev=1e-6)
    table = lookup_table_constant(REGISTER_TOKENS, num_register_tokens, embed_dim, initializer, tokens)  # fmt: skip
    broadcasted = broadcast_multi_tokens(table, tokens)
    return ops.concatenate([tokens[:, :1], broadcasted, tokens[:, 1:]], axis=1)


def broadcast_multi_tokens(tokens, reference):
    num = tokens.shape[1]
    # zeros_like carries reference's dynamic batch (see broadcast_single_token)
    return ops.zeros_like(reference[:, :num, :]) + tokens


def build_transformer_blocks(tokens, embed_dim, depth, num_heads, init_values, feedforward_layer):  # fmt: skip
    drop_rates = compute_drop_path_rates(0.0, depth, False)
    normalization = partial(layers.LayerNormalization, epsilon=1e-6)
    activation = "silu" if feedforward_layer == "swiglu" else "gelu"
    keys = ("dim", "num_heads", "use_qkv_bias", "use_projection_bias", "attention_dropout_rate", "feedforward_layer", "mlp_ratio", "use_feedforward_bias", "activation", "drop_rate", "init_values", "normalization_layer")  # fmt: skip
    values = (embed_dim, num_heads, True, True, 0.0, feedforward_layer, 4.0, True, activation, 0.0, init_values, normalization)  # fmt: skip
    apply_block = partial(block, **dict(zip(keys, values)))
    for index in range(depth):
        tokens = apply_block(tokens, drop_path=drop_rates[index], name=f"block_{index}")  # fmt: skip
    return tokens


def compute_drop_path_rates(drop_path_rate, depth, drop_path_uniform):
    uniform = [drop_path_rate] * depth
    linear = np.linspace(0.0, drop_path_rate, depth).tolist()
    return uniform if drop_path_uniform else linear


def DINOV2Small(patch_size=14, num_register_tokens=0, img_size=518, init_values=1e-5, name=None):  # fmt: skip
    return DINOV2VIT(img_size, patch_size, 384, 12, 6, num_register_tokens, init_values, name=name)  # fmt: skip


def DINOV2Base(patch_size=14, num_register_tokens=0, img_size=518, init_values=1e-5, name=None):  # fmt: skip
    return DINOV2VIT(img_size, patch_size, 768, 12, 12, num_register_tokens, init_values, name=name)  # fmt: skip


def DINOV2Large(patch_size=14, num_register_tokens=0, img_size=518, init_values=1e-5, name=None):  # fmt: skip
    return DINOV2VIT(img_size, patch_size, 1024, 24, 16, num_register_tokens, init_values, name=name)  # fmt: skip


def DINOV2Giant2(patch_size=14, num_register_tokens=0, img_size=518, init_values=1e-5, feedforward_layer="swiglu", name=None):  # fmt: skip
    return DINOV2VIT(img_size, patch_size, 1536, 40, 24, num_register_tokens, init_values, feedforward_layer, name=name)  # fmt: skip
