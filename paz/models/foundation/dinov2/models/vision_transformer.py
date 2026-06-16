from functools import partial

import numpy as np
import keras
from keras import Input, Model, layers, ops, initializers

from paz.models.foundation.dinov2.layers import (
    build_patch_embedding,
    block,
    AttentionArgs,
    FFNArgs,
)

CLS_TOKEN = "cls_token"
POS_EMBED = "pos_embed"
REGISTER_TOKENS = "register_tokens"
PATCH_EMBED = "patch_embed"
FINAL_NORM = "norm"


def DINOV2VIT(img_size, patch_size, embed_dim, depth, num_heads, num_reg_tokens=0, init_values=1e-5, FFN_layer="mlp", name=None,): # fmt: skip
    pixels = Input((img_size, img_size, 3), name="pixels")
    patches = build_patch_embedding(pixels, patch_size, embed_dim, PATCH_EMBED)
    num_patches = (img_size // patch_size) ** 2
    tokens = build_token_stream(patches, embed_dim, num_patches, num_reg_tokens)
    keys = (
        "mlp_ratio",
        "use_qkv_bias",
        "use_proj_bias",
        "use_FFN_bias",
        "init_values",
        "drop_path",
        "drop_path_uniform",
        "FFN_layer",
    )
    values = (4.0, True, True, True, init_values, 0.0, False, FFN_layer)
    config = dict(zip(keys, values))
    args = (tokens, embed_dim, depth, num_heads, config)
    features = build_transformer_blocks(*args)
    normalized = apply_final_norm(features)
    cls_output = take_first_token(normalized)
    return Model(pixels, cls_output, name=name)


def apply_final_norm(features):
    layer = layers.LayerNormalization(epsilon=1e-6, name=FINAL_NORM)
    return layer(features)


def take_first_token(features):
    return features[:, 0]


def build_token_stream(patches, embed_dim, num_patches, num_register_tokens):
    init_cls = initializers.RandomNormal(stddev=1e-6)
    init_pos = initializers.TruncatedNormal(stddev=0.02)
    cls = lookup_table_constant(CLS_TOKEN, 1, embed_dim, init_cls, patches)
    tokens = prepend_cls_token(patches, cls)
    pos_anchor = tokens
    pos = lookup_table_constant(
        POS_EMBED, num_patches + 1, embed_dim, init_pos, pos_anchor
    )
    tokens = add_positional_embedding(tokens, pos)
    return insert_register_tokens(tokens, num_register_tokens, embed_dim)


def lookup_table_constant(name, num, embed_dim, initializer, anchor):
    args = dict(embeddings_initializer=initializer, name=name)
    embedding = layers.Embedding(num, embed_dim, **args)
    indices = build_indices(num, anchor, name)
    return embedding(indices)


def build_indices(num, anchor, name):
    layer = layers.Lambda(
        arange_indices,
        arguments={"num": num},
        output_shape=(num,),
        name=f"{name}_indices",
    )
    return layer(anchor)


@keras.saving.register_keras_serializable(package="paz_dinov2")
def arange_indices(reference, num=0):
    return ops.arange(num, dtype="int32")


def prepend_cls_token(patches, cls):
    broadcasted = broadcast_single_token(cls, patches)
    return ops.concatenate([broadcasted, patches], axis=1)


def broadcast_single_token(token, reference):
    zeros = reference[:, :1, :] * 0.0
    return zeros + token


def add_positional_embedding(tokens, pos_table):
    expanded = pos_table
    return tokens + expanded


def insert_register_tokens(tokens, num_register_tokens, embed_dim):
    if num_register_tokens == 0:
        return tokens
    return apply_register_tokens(tokens, num_register_tokens, embed_dim)


def apply_register_tokens(tokens, num, embed_dim):
    initializer = initializers.RandomNormal(stddev=1e-6)
    table = lookup_table_constant(REGISTER_TOKENS, num, embed_dim, initializer, tokens)
    broadcasted = broadcast_multi_tokens(table, tokens)
    return ops.concatenate([tokens[:, :1], broadcasted, tokens[:, 1:]], axis=1)


def broadcast_multi_tokens(tokens, reference):
    num = tokens.shape[1]
    zeros = reference[:, :num, :] * 0.0
    return zeros + tokens


def build_transformer_blocks(tokens, embed_dim, depth, num_heads, config):
    rate, uniform = config["drop_path"], config["drop_path_uniform"]
    drop_rates = compute_drop_path_rates(rate, depth, uniform)
    normalization = partial(layers.LayerNormalization, epsilon=1e-6)
    for i in range(depth):
        args = (embed_dim, num_heads, config, drop_rates[i], normalization, i)
        tokens = block(tokens, **build_block_kwargs(*args))
    return tokens


def build_block_kwargs(embed_dim, num_heads, config, drop_rate, normalization, index):
    mlp_ratio, use_FFN_bias = config["mlp_ratio"], config["use_FFN_bias"]
    qkv_bias, proj_bias = config["use_qkv_bias"], config["use_proj_bias"]
    FFN_layer, init_values = config["FFN_layer"], config["init_values"]
    attention = AttentionArgs(num_heads, qkv_bias, proj_bias, 0.0)
    activation = "silu" if FFN_layer == "swiglu" else "gelu"
    FFN = FFNArgs(FFN_layer, mlp_ratio, use_FFN_bias, activation)
    return dict(
        dim=embed_dim,
        attention=attention,
        FFN=FFN,
        drop_rate=0.0,
        init_values=init_values,
        drop_path=drop_rate,
        normalization_layer=normalization,
        name=f"block_{index}",
    )


def compute_drop_path_rates(drop_path_rate, depth, drop_path_uniform):
    uniform = [drop_path_rate] * depth
    linear = np.linspace(0.0, drop_path_rate, depth).tolist()
    return uniform if drop_path_uniform else linear


def DINOV2Small(patch_size=14, num_reg_tokens=0, img_size=518, init_values=1e-5, name=None):  # fmt: skip
    return DINOV2VIT(
        img_size, patch_size, 384, 12, 6, num_reg_tokens, init_values, name=name
    )


def DINOV2Base(patch_size=14, num_reg_tokens=0, img_size=518, init_values=1e-5, name=None):  # fmt: skip
    return DINOV2VIT(
        img_size, patch_size, 768, 12, 12, num_reg_tokens, init_values, name=name
    )


def DINOV2Large(patch_size=14, num_reg_tokens=0, img_size=518, init_values=1e-5, name=None):  # fmt: skip
    return DINOV2VIT(
        img_size, patch_size, 1024, 24, 16, num_reg_tokens, init_values, name=name
    )


def DINOV2Giant2(patch_size=14, num_reg_tokens=0, img_size=518, init_values=1e-5, FFN_layer="swiglu", name=None):  # fmt: skip
    return DINOV2VIT(
        img_size,
        patch_size,
        1536,
        40,
        24,
        num_reg_tokens,
        init_values,
        FFN_layer,
        name=name,
    )
