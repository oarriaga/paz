from functools import partial

import numpy as np
import keras
from keras import Input, Model, layers, ops, initializers

from paz.models.foundation.dinov2_legacy.layers import build_patch_embedding, block

CLS_TOKEN = "cls_token"
POS_EMBED = "pos_embed"
REGISTER_TOKENS = "register_tokens"
PATCH_EMBED = "patch_embed"
FINAL_NORM = "norm"


def DINOV2VIT(img_size, patch_size, embed_dim, depth, num_heads, num_register_tokens=0, init_values=1e-5, FFN_layer="mlp", name=None):  # fmt: skip
    pixels = Input((img_size, img_size, 3), name="pixels")
    patches = build_patch_embedding(pixels, patch_size, embed_dim, PATCH_EMBED)
    num_patches = (img_size // patch_size) ** 2
    tokens = build_token_stream(patches, embed_dim, num_patches, num_register_tokens)  # fmt: skip
    features = build_transformer_blocks(tokens, embed_dim, depth, num_heads, init_values, FFN_layer)  # fmt: skip
    normalized = layers.LayerNormalization(epsilon=1e-6, name=FINAL_NORM)(features)
    CLS_output = take_first_token(normalized)
    return Model(pixels, CLS_output, name=name)


def take_first_token(features):
    return features[:, 0]


def build_token_stream(patches, embed_dim, num_patches, num_register_tokens):
    init_CLS = initializers.RandomNormal(stddev=1e-6)
    init_POS = initializers.TruncatedNormal(stddev=0.02)
    CLS = lookup_table_constant(CLS_TOKEN, 1, embed_dim, init_CLS, patches)
    tokens = prepend_CLS_token(patches, CLS)
    POS = lookup_table_constant(POS_EMBED, num_patches + 1, embed_dim, init_POS, tokens)  # fmt: skip
    tokens = tokens + POS
    return insert_register_tokens(tokens, num_register_tokens, embed_dim)


def lookup_table_constant(name, num, embed_dim, initializer, anchor):
    kwargs = dict(embeddings_initializer=initializer, name=name)
    indices = Indices(num, anchor, name)
    return layers.Embedding(num, embed_dim, **kwargs)(indices)


def Indices(num, anchor, name):
    return layers.Lambda(
        arange_indices,
        arguments={"num": num},
        output_shape=(num,),
        name=f"{name}_indices",
    )(anchor)


@keras.saving.register_keras_serializable(package="paz_dinov2")
def arange_indices(reference, num=0):
    # registered so ported .keras files using it in a Lambda can deserialize
    return ops.arange(num, dtype="int32")


def prepend_CLS_token(patches, CLS):
    broadcasted = broadcast_single_token(CLS, patches)
    return ops.concatenate([broadcasted, patches], axis=1)


def broadcast_single_token(token, reference):
    # zeros_like carries reference's dynamic batch; ops.broadcast_to with a
    # symbolic batch (ops.shape(reference)[0]) fails at graph-build time
    return ops.zeros_like(reference[:, :1, :]) + token


def insert_register_tokens(tokens, num_register_tokens, embed_dim):
    if num_register_tokens != 0:
        initializer = initializers.RandomNormal(stddev=1e-6)
        table = lookup_table_constant(REGISTER_TOKENS, num_register_tokens, embed_dim, initializer, tokens)  # fmt: skip
        broadcasted = broadcast_multi_tokens(table, tokens)
        tokens = ops.concatenate([tokens[:, :1], broadcasted, tokens[:, 1:]], axis=1)
    return tokens


def broadcast_multi_tokens(tokens, reference):
    num = tokens.shape[1]
    # zeros_like carries reference's dynamic batch (see broadcast_single_token)
    return ops.zeros_like(reference[:, :num, :]) + tokens


def build_transformer_blocks(tokens, embed_dim, depth, num_heads, init_values, FFN_layer):  # fmt: skip
    drop_rates = compute_drop_path_rates(0.0, depth, False)
    normalization = partial(layers.LayerNormalization, epsilon=1e-6)
    activation = "silu" if FFN_layer == "swiglu" else "gelu"
    keys = ("dim", "num_heads", "use_QKV_bias", "use_projection_bias", "attention_dropout_rate", "FFN_layer", "MLP_ratio", "use_FFN_bias", "activation", "drop_rate", "init_values", "normalization_layer")  # fmt: skip
    values = (embed_dim, num_heads, True, True, 0.0, FFN_layer, 4.0, True, activation, 0.0, init_values, normalization)  # fmt: skip
    apply_block = partial(block, **dict(zip(keys, values)))
    for depth_arg in range(depth):
        name = f"block_{depth_arg}"
        tokens = apply_block(tokens, drop_path=drop_rates[depth_arg], name=name)
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


def DINOV2Giant2(patch_size=14, num_register_tokens=0, img_size=518, init_values=1e-5, FFN_layer="swiglu", name=None):  # fmt: skip
    return DINOV2VIT(img_size, patch_size, 1536, 40, 24, num_register_tokens, init_values, FFN_layer, name=name)  # fmt: skip
