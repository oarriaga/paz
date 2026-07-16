"""Hiera multiscale image backbone for SAM 2 (channels-last, functional).

Reuses the shared attention scaling from ``paz.models.transformers`` and the
GELU feedforward. Query pooling, window attention, and the interpolated
windowed positional embedding are SAM-specific and stay local.
"""
import keras
import numpy as np
from keras import ops
from keras.layers import Conv2D, Dense, LayerNormalization
from keras.layers import MaxPooling2D, ZeroPadding2D, Layer

from paz.models.transformers.attention import compute_attention, kernel
from paz.models.transformers import feedforward
from paz.models.foundation.sam2 import configuration as config
from paz.models.foundation.sam2.windows import window_partition
from paz.models.foundation.sam2.windows import window_unpartition


@keras.saving.register_keras_serializable(package="paz")
class HieraPositionEmbedding(Layer):
    def __init__(self, hidden_size, background_size, window, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.background_size = tuple(background_size)
        self.window = window

    def build(self, input_shape):
        background = (*self.background_size, self.hidden_size)
        window = (self.window, self.window, self.hidden_size)
        self.background = self.add_weight(
            name="background", shape=background, initializer="zeros")
        self.window_embed = self.add_weight(
            name="window", shape=window, initializer="zeros")

    def call(self, x):
        height, width = x.shape[1], x.shape[2]
        rows = bicubic_resize_matrix(self.background_size[0], height)
        columns = bicubic_resize_matrix(self.background_size[1], width)
        background = ops.einsum("hn,nmc->hmc", rows, self.background)
        background = ops.einsum("wm,hmc->hwc", columns, background)
        tiles = (height // self.window, width // self.window, 1)
        window = ops.tile(self.window_embed, tiles)
        return x + background + window

    def get_config(self):
        arguments = dict(hidden_size=self.hidden_size, window=self.window,
                         background_size=self.background_size)
        return {**super().get_config(), **arguments}


def bicubic_resize_matrix(source, target, coefficient=-0.75):
    scale = source / target
    matrix = np.zeros((target, source), np.float32)
    for row in range(target):
        center = (row + 0.5) * scale - 0.5
        left = int(np.floor(center))
        fraction = center - left
        for offset in (-1, 0, 1, 2):
            column = min(max(left + offset, 0), source - 1)
            matrix[row, column] += cubic_weight(fraction - offset, coefficient)
    return matrix


def cubic_weight(distance, coefficient):
    distance = abs(distance)
    if distance <= 1:
        return ((coefficient + 2) * distance - (coefficient + 3)) \
            * distance * distance + 1
    if distance < 2:
        return coefficient * (((distance - 5) * distance + 8) * distance - 4)
    return 0.0


def build(images, model_config):
    tokens = build_patch_embedding(images, model_config.embed_dim)
    tokens = HieraPositionEmbedding(
        model_config.embed_dim, model_config.background_size,
        model_config.window_spec[0], name="trunk_pos_embed")(tokens)
    specifications, stage_ends = build_block_specifications(model_config)
    outputs = []
    for specification in specifications:
        tokens = apply_block(tokens, *specification)
        if specification[0] in stage_ends:
            outputs.append(tokens)
    return outputs


def build_patch_embedding(images, hidden_size):
    padded = ZeroPadding2D(3, name="patch_embed_pad")(images)
    kwargs = dict(strides=4, padding="valid", kernel_initializer=kernel(),
                  name="patch_embed_proj")
    return Conv2D(hidden_size, 7, **kwargs)(padded)


def build_block_specifications(model_config):
    stages = model_config.stages
    global_blocks = set(model_config.global_attention_blocks)
    stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
    starts = [end + 1 for end in stage_ends[:-1]]
    pool_blocks = starts[:config.QUERY_POOL_STAGES]
    embed_dim = model_config.embed_dim
    num_heads = model_config.num_heads
    current_stage = 1
    specifications = []
    for index in range(sum(stages)):
        dim, dim_out = embed_dim, embed_dim
        window = model_config.window_spec[current_stage - 1]
        if index in global_blocks:
            window = 0
        if (index - 1) in stage_ends:
            dim_out = embed_dim * config.DIM_MUL
            num_heads = num_heads * config.HEAD_MUL
            current_stage = current_stage + 1
        pool = index in pool_blocks
        name = f"trunk_block_{index}"
        specifications.append((index, dim, dim_out, num_heads, window, pool,
                               name))
        embed_dim = dim_out
    return specifications, stage_ends


def apply_block(x, index, dim, dim_out, num_heads, window, pool, name):
    shortcut = x
    normed = LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(x)
    if dim != dim_out:
        projected = Dense(dim_out, kernel_initializer=kernel(),
                          name=f"{name}_proj")(normed)
        shortcut = pool_spatial(projected) if pool else projected
    height, width = normed.shape[1], normed.shape[2]
    padded_size = (height, width)
    if window > 0:
        normed, padded_size = window_partition(normed, window)
    attended = apply_attention(normed, dim_out, num_heads, pool, f"{name}_attn")
    if pool:
        window = window // config.QUERY_STRIDE[0]
        height, width = shortcut.shape[1], shortcut.shape[2]
        padded_size = padded_after_pool(height, width, window)
    if window > 0:
        attended = window_unpartition(attended, window, padded_size,
                                      (height, width))
    tokens = ops.add(shortcut, attended)
    return apply_feedforward(tokens, dim_out, name)


def apply_attention(x, dim_out, num_heads, pool, name):
    height, width = x.shape[1], x.shape[2]
    head_dim = dim_out // num_heads
    fused = Dense(dim_out * 3, kernel_initializer=kernel(),
                  name=f"{name}_qkv")(x)
    fused = ops.reshape(fused, (-1, height * width, 3, num_heads, head_dim))
    query, key, value = fused[:, :, 0], fused[:, :, 1], fused[:, :, 2]
    if pool:
        query = ops.reshape(query, (-1, height, width, dim_out))
        query = pool_spatial(query)
        height, width = height // 2, width // 2
        query = ops.reshape(query, (-1, height * width, num_heads, head_dim))
    query = ops.transpose(query, (0, 2, 1, 3))
    key = ops.transpose(key, (0, 2, 1, 3))
    value = ops.transpose(value, (0, 2, 1, 3))
    context = compute_attention(query, key, value)
    context = ops.transpose(context, (0, 2, 1, 3))
    context = ops.reshape(context, (-1, height, width, dim_out))
    return Dense(dim_out, kernel_initializer=kernel(),
                 name=f"{name}_proj")(context)


def apply_feedforward(tokens, dim_out, name):
    normed = LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(tokens)
    inner = int(dim_out * config.MLP_RATIO)
    names = f"{name}_mlp_fc1", f"{name}_mlp_fc2"
    forwarded = feedforward.gelu(normed, inner, dim_out, *names)
    return ops.add(tokens, forwarded)


def pool_spatial(x):
    return MaxPooling2D(config.QUERY_STRIDE, config.QUERY_STRIDE,
                        padding="valid")(x)


def padded_after_pool(height, width, window):
    pad_h = (window - height % window) % window
    pad_w = (window - width % window) % window
    return (height + pad_h, width + pad_w)
