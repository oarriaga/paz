"""Swin Transformer backbone.

Tokens attend only inside square windows, and every second block shifts the
windows by half their size so that information crosses window borders. Each
stage halves the resolution by merging ``2 x 2`` patches. Image sides must be
multiples of 32 so that every merge sees an even grid.
"""
import numpy as np
from keras import ops
from keras.layers import Conv2D, Dense

from paz.models.detection.rf_detr.decoder import build_table, normalize
from paz.models.foundation.sam2.windows import window_partition
from paz.models.foundation.sam2.windows import window_unpartition
from paz.models.transformers import feedforward
from paz.models.transformers.attention import merge_attention_heads
from paz.models.transformers.attention import project_query_key_value
from paz.models.transformers.attention import split_query_key_value


def build(images, embed_dim, depths, num_heads, window_size):
    """Returns the feature map of every stage after the first."""
    args = depths[0], num_heads[0], window_size, "backbone_stages_0"
    x = apply_stage(embed_patches(images, embed_dim), *args)
    features = []
    for stage in range(1, len(depths)):
        x = merge_patches(x, f"backbone_stages_{stage - 1}_downsample")
        args = depths[stage], num_heads[stage], window_size
        x = apply_stage(x, *args, f"backbone_stages_{stage}")
        features.append(normalize(x, f"backbone_norm{stage}"))
    return features


def embed_patches(images, embed_dim):
    x = Conv2D(embed_dim, 4, 4, name="backbone_patch_embed_projection")(images)
    return normalize(x, "backbone_patch_embed_norm")


def apply_stage(x, depth, num_heads, window_size, name):
    for block in range(depth):
        shift = (block % 2) * (window_size // 2)
        args = num_heads, window_size, shift, f"{name}_blocks_{block}"
        x = apply_block(x, *args)
    return x


def apply_block(x, num_heads, window_size, shift, name):
    hidden_size = x.shape[-1]
    args = num_heads, window_size, shift, f"{name}_attn_w_msa"
    x = x + attend_windows(normalize(x, f"{name}_norm1"), *args)
    names = f"{name}_ffn_layers_0_0", f"{name}_ffn_layers_1"
    args = 4 * hidden_size, hidden_size, *names
    return x + feedforward.gelu(normalize(x, f"{name}_norm2"), *args)


def attend_windows(x, num_heads, window_size, shift, name):
    height, width, hidden_size = x.shape[1:]
    shifted = ops.roll(pad_to_windows(x, window_size), (-shift, -shift), (1, 2))
    windows, padded_size = window_partition(shifted, window_size)
    tokens = ops.reshape(windows, (-1, window_size**2, hidden_size))
    bias = build_position_bias(tokens, window_size, num_heads, name)
    bias = bias + build_shift_mask(*padded_size, window_size, shift)
    tokens = attend_tokens(tokens, bias, num_heads, name)
    windows = ops.reshape(tokens, (-1, window_size, window_size, hidden_size))
    shifted = window_unpartition(windows, window_size, *[padded_size] * 2)
    return ops.roll(shifted, (shift, shift), (1, 2))[:, :height, :width]


def pad_to_windows(x, window_size):
    pad_height = -x.shape[1] % window_size
    pad_width = -x.shape[2] % window_size
    return ops.pad(x, [[0, 0], [0, pad_height], [0, pad_width], [0, 0]])


def attend_tokens(tokens, bias, num_heads, name):
    """Self-attention with ``bias`` added to the scores of every window."""
    hidden_size = tokens.shape[-1]
    head_dim = hidden_size // num_heads
    fused = project_query_key_value(tokens, hidden_size, True, name)
    query, key, value = split_query_key_value(fused, num_heads, head_dim)
    scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
    scores = ops.reshape(scores * head_dim**-0.5, (-1, *bias.shape)) + bias
    weights = ops.softmax(ops.reshape(scores, (-1, *bias.shape[1:])), axis=-1)
    context = merge_attention_heads(ops.matmul(weights, value))
    return Dense(hidden_size, name=f"{name}_proj")(context)


def build_position_bias(tokens, window_size, num_heads, name):
    """Learned score offset for every pair of cells, by relative position."""
    count = (2 * window_size - 1) ** 2
    name = f"{name}_relative_position_bias_table"
    table = build_table(tokens, count, num_heads, name)[0]
    index = build_position_index(window_size)
    return ops.transpose(ops.take(table, index, axis=0), (2, 0, 1))


def build_position_index(window_size):
    """Table row of every cell pair, in the layout of the published weights."""
    step = 2 * window_size - 1
    rows = np.arange(window_size)[:, None] * step
    cells = (rows + np.arange(window_size)[None, :]).reshape(1, -1)
    return (cells + cells.T)[:, ::-1]


def build_shift_mask(height, width, window_size, shift):
    """Blocks attention between cells that only the shift made neighbours."""
    rows = label_axis(height, window_size, shift)
    labels = 3 * rows[:, None] + label_axis(width, window_size, shift)
    shape = height // window_size, window_size, width // window_size, -1
    windows = labels.reshape(shape).transpose(0, 2, 1, 3)
    windows = windows.reshape(-1, window_size**2)
    separated = windows[:, :, None] != windows[:, None, :]
    return -100.0 * separated[:, None].astype("float32")


def label_axis(length, window_size, shift):
    positions = np.arange(length)
    wrapped = positions >= length - shift
    return (positions >= length - window_size) + wrapped.astype(int)


def merge_patches(x, name):
    """Stacks each ``2 x 2`` patch on the channel axis and halves its width."""
    height, width, channels = x.shape[1:]
    x = ops.reshape(x, (-1, height // 2, 2, width // 2, 2, channels))
    x = ops.transpose(x, (0, 1, 3, 5, 2, 4))
    x = ops.reshape(x, (-1, height // 2, width // 2, 4 * channels))
    x = normalize(x, f"{name}_norm")
    return Dense(2 * channels, use_bias=False, name=f"{name}_reduction")(x)
