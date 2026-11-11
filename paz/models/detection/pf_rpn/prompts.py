"""Pseudo-text tokens computed from the image, replacing a text prompt.

Grounding DINO scores boxes against text tokens. PF-RPN builds those tokens
from the image itself: every feature level proposes a set of tokens, a router
mixes the proposals of its preferred levels, and the token that stands for
"object" is then pulled towards the image cells that resemble it.
"""
import numpy as np
from keras import Input, Model, ops
from keras.layers import Dense

from paz.models.detection.rf_detr.decoder import build_table, normalize
from paz.models.transformers import feedforward
from paz.models.transformers.attention import masked_attend


def build(memory, grids, num_tokens=3, num_selected=2, threshold=0.3,
          num_iterations=3):
    hidden_size = memory.shape[-1]
    pool = build_pool(hidden_size, 32)
    context_decoder = build_context_decoder(hidden_size, 4, 3)
    args = memory, num_tokens, hidden_size, "learnable_text_embedding"
    queries = build_table(*args)
    # The published table holds one position per cell, up to 30000 cells.
    if grids[0][0] * grids[0][1] > 30000:
        raise ValueError(f"A level holds at most 30000 cells: {grids[0]}")
    args = memory, 30001, hidden_size, "vis_proj_positional_embedding"
    positions = build_table(*args)
    summaries, proposals = [], []
    for cells in split_levels(memory, grids):
        tokens = prepend_mean(cells)
        pooled = pool(tokens + positions[:, :tokens.shape[1]])
        summaries.append(pooled[:, :1])
        proposals.append(context_decoder([queries, pooled]))
    weights = route(ops.concatenate(summaries, axis=1), num_selected)
    texts = ops.sum(ops.stack(proposals, axis=1) * weights[..., None], axis=1)
    return enhance(texts, memory, grids, threshold, num_iterations)


def split_levels(memory, grids):
    levels, start = [], 0
    for height, width in grids:
        levels.append(memory[:, start:start + height * width])
        start = start + height * width
    return levels


def prepend_mean(cells):
    mean = ops.mean(cells, axis=1, keepdims=True)
    return ops.concatenate([mean, cells], axis=1)


def build_pool(hidden_size, num_heads):
    """Self-attention over one level; shared by all levels."""
    tokens = Input((None, hidden_size))
    query = Dense(hidden_size, name="vis_proj_query")(tokens)
    key = Dense(hidden_size, name="vis_proj_key")(tokens)
    value = Dense(hidden_size, name="vis_proj_value")(tokens)
    parts = query, key, value
    heads = [ops.split(part, num_heads, axis=-1) for part in parts]
    context = [attend_head(*head) for head in zip(*heads)]
    context = ops.concatenate(context, axis=-1)
    pooled = Dense(hidden_size, name="vis_proj_attention_output")(context)
    return Model(tokens, pooled, name="vis_proj")


def attend_head(query, key, value):
    """Attention of one head as one op, so its scores are freed right away.

    A level holds thousands of cells: the scores of all heads at once take
    5 GB at 640 x 640 pixels, and an eager model keeps every op output alive.
    """
    heads = [ops.expand_dims(part, axis=2) for part in (query, key, value)]
    return ops.dot_product_attention(*heads)[:, :, 0]


def build_context_decoder(hidden_size, num_heads, num_layers):
    """Lets the token queries read one pooled level; shared by all levels."""
    queries = Input((None, hidden_size))
    memory = Input((None, hidden_size))
    visual = normalize(memory, "meta_net_memory_proj_0")
    visual = Dense(hidden_size, name="meta_net_memory_proj_1")(visual)
    visual = normalize(visual, "meta_net_memory_proj_2")
    x = normalize(queries, "meta_net_text_proj_0")
    x = Dense(hidden_size, name="meta_net_text_proj_1")(x)
    for layer in range(num_layers):
        name = f"meta_net_decoder_{layer}"
        x = apply_context_layer(x, visual, num_heads, name)
    x = normalize(x, "meta_net_out_proj_0")
    x = Dense(hidden_size, name="meta_net_out_proj_1")(x)
    return Model([queries, memory], x, name="meta_net")


def apply_context_layer(x, memory, num_heads, name):
    hidden_size = x.shape[-1]
    args = None, num_heads, hidden_size // num_heads, 0.0
    query = normalize(x, f"{name}_norm1")
    x = x + masked_attend(query, query, *args, f"{name}_self_attn")
    query = normalize(x, f"{name}_norm2")
    x = x + masked_attend(query, memory, *args, f"{name}_cross_attn")
    args = 4 * hidden_size, hidden_size, f"{name}_mlp_0", f"{name}_mlp_3"
    return x + feedforward.gelu(normalize(x, f"{name}_norm3"), *args)


def route(summaries, num_selected):
    """Softmax weight per level, zero outside the ``num_selected`` best."""
    logits = Dense(1, name="router")(summaries)
    best = ops.top_k(logits[..., 0], num_selected)[0]
    lowest = ops.min(best, axis=-1, keepdims=True)[..., None]
    return ops.softmax(ops.where(logits >= lowest, logits, -np.inf), axis=1)


def enhance(texts, memory, grids, threshold, num_iterations):
    """Adds to the object token the mean of the cells that resemble it."""
    # Upstream trains token 1 as the only positive one, so only it is enhanced.
    token = texts[:, 1:2]
    for iteration in range(num_iterations):
        for cells in split_levels(memory, grids):
            token = token + average_similar(cells, token, threshold)
    return ops.concatenate([texts[:, :1], token, texts[:, 2:]], axis=1)


def average_similar(cells, token, threshold):
    direction = ops.transpose(ops.normalize(token, axis=-1), (0, 2, 1))
    similarity = ops.matmul(ops.normalize(cells, axis=-1), direction)
    mask = ops.cast(similarity > threshold, cells.dtype)
    count = ops.maximum(ops.sum(mask, axis=1, keepdims=True), 1e-6)
    return ops.sum(cells * mask, axis=1, keepdims=True) / count
