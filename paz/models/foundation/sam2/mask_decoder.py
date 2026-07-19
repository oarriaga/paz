"""SAM 2 mask decoder: two-way transformer, upscaling, and prediction heads.

Reuses ``compute_attention`` from ``paz.models.transformers`` for scaled
dot-product attention. Output tokens (object-score, IoU, and four mask
tokens) attend to the image embedding and back; a hypernetwork projects the
mask tokens onto the upscaled features to produce four low-resolution masks.
"""
from keras import Input, Model, ops
from keras.layers import Conv2DTranspose, Dense, LayerNormalization, Reshape

from paz.models.transformers.attention import compute_attention
from paz.models.foundation.sam2.configuration import PROMPT_EMBED_DIM
from paz.models.foundation.sam2.layers import BroadcastTokens
from paz.models.foundation.sam2.prompt_encoder import GRID

NUM_HEADS = 8
NUM_MASK_TOKENS = 4
TOKENS = 6


def build(name="sam2_mask_decoder"):
    embed = Input((GRID, GRID, PROMPT_EMBED_DIM), name="image_embed")
    high_res_0 = Input((256, 256, 32), name="high_res_0")
    high_res_1 = Input((128, 128, 64), name="high_res_1")
    sparse = Input((None, PROMPT_EMBED_DIM), name="sparse")
    dense = Input((GRID, GRID, PROMPT_EMBED_DIM), name="dense")
    image_pe = Input((GRID, GRID, PROMPT_EMBED_DIM), name="image_pe")

    tokens = build_output_tokens(sparse)
    source = flatten(ops.add(embed, dense))
    positions = flatten(image_pe)
    queries, keys = two_way_transformer(source, positions, tokens)
    upscaled = upscale(reshape_to_grid(keys), high_res_0, high_res_1)
    masks = hypernetwork(queries[:, 2:TOKENS], upscaled)
    iou_args = queries[:, 1], 256, NUM_MASK_TOKENS, 3, "iou_prediction_head"
    iou = mlp(*iou_args, sigmoid=True)
    obj = mlp(queries[:, 0], 256, 1, 3, "pred_obj_score_head")
    inputs = (embed, high_res_0, high_res_1, sparse, dense, image_pe)
    return Model(inputs, (masks, iou, obj), name=name)


def build_output_tokens(sparse):
    obj = output_token(1, "obj_score_token", sparse)
    iou = output_token(1, "iou_token", sparse)
    mask = output_token(NUM_MASK_TOKENS, "mask_tokens", sparse)
    return ops.concatenate([obj, iou, mask, sparse], axis=1)


def output_token(count, name, reference):
    return BroadcastTokens(count, PROMPT_EMBED_DIM, name=name)(reference)


def two_way_transformer(source, positions, tokens):
    queries, keys = tokens, source
    for index in range(2):
        queries, keys = two_way_block(queries, keys, tokens, positions, index)
    query = ops.add(queries, tokens)
    key = ops.add(keys, positions)
    attended = attention(query, key, keys, 2, "twoway_final_attn")
    queries = ops.add(queries, attended)
    queries = normalize(queries, "twoway_norm_final")
    return queries, keys


def two_way_block(queries, keys, query_pe, key_pe, index):
    name = f"twoway_{index}"
    if index == 0:
        queries = attention(queries, queries, queries, 1, f"{name}_self")
    else:
        query = ops.add(queries, query_pe)
        attended = attention(query, query, queries, 1, f"{name}_self")
        queries = ops.add(queries, attended)
    queries = normalize(queries, f"{name}_norm1")
    query = ops.add(queries, query_pe)
    key = ops.add(keys, key_pe)
    attended = attention(query, key, keys, 2, f"{name}_cross_t2i")
    queries = normalize(ops.add(queries, attended), f"{name}_norm2")
    forwarded = mlp(queries, 2048, PROMPT_EMBED_DIM, 2, f"{name}_mlp")
    queries = normalize(ops.add(queries, forwarded), f"{name}_norm3")
    query = ops.add(queries, query_pe)
    key = ops.add(keys, key_pe)
    attended = attention(key, query, queries, 2, f"{name}_cross_i2t")
    keys = normalize(ops.add(keys, attended), f"{name}_norm4")
    return queries, keys


def attention(query, key, value, downsample, name):
    internal = PROMPT_EMBED_DIM // downsample
    head_dim = internal // NUM_HEADS
    query = split_heads(Dense(internal, name=f"{name}_q")(query), head_dim)
    key = split_heads(Dense(internal, name=f"{name}_k")(key), head_dim)
    value = split_heads(Dense(internal, name=f"{name}_v")(value), head_dim)
    context = compute_attention(query, key, value)
    context = merge_heads(context, internal)
    return Dense(PROMPT_EMBED_DIM, name=f"{name}_out")(context)


def split_heads(x, head_dim):
    x = Reshape((-1, NUM_HEADS, head_dim))(x)
    return ops.transpose(x, (0, 2, 1, 3))


def merge_heads(context, internal):
    context = ops.transpose(context, (0, 2, 1, 3))
    return Reshape((-1, internal))(context)


def upscale(source, high_res_0, high_res_1):
    x = Conv2DTranspose(64, 2, strides=2, name="output_upscaling_0")(source)
    x = normalize_channels(ops.add(x, high_res_1), "output_upscaling_1")
    x = ops.gelu(x, approximate=False)
    x = Conv2DTranspose(32, 2, strides=2, name="output_upscaling_3")(x)
    x = ops.gelu(ops.add(x, high_res_0), approximate=False)
    return x


def hypernetwork(mask_tokens, upscaled):
    projections = []
    for index in range(NUM_MASK_TOKENS):
        token = mask_tokens[:, index]
        name = f"output_hypernetworks_mlps_{index}"
        projections.append(mlp(token, 256, 32, 3, name))
    weights = ops.stack(projections, axis=1)
    return ops.einsum("bkc,bhwc->bkhw", weights, upscaled)


def mlp(x, hidden, output, layers, name, sigmoid=False):
    for index in range(layers - 1):
        x = Dense(hidden, activation="relu", name=f"{name}_layers_{index}")(x)
    x = Dense(output, name=f"{name}_layers_{layers - 1}")(x)
    return ops.sigmoid(x) if sigmoid else x


def flatten(x):
    return ops.reshape(x, (-1, GRID * GRID, PROMPT_EMBED_DIM))


def reshape_to_grid(x):
    return ops.reshape(x, (-1, GRID, GRID, PROMPT_EMBED_DIM))


def normalize(x, name):
    return LayerNormalization(epsilon=1e-5, name=name)(x)


def normalize_channels(x, name):
    return LayerNormalization(axis=-1, epsilon=1e-6, name=name)(x)
