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

from paz.models.transformers.attention import compute_attention
from paz.models.transformers import feedforward
from paz.models.foundation.sam2 import configuration as cfg
from paz.models.foundation.sam2.windows import window_partition
from paz.models.foundation.sam2.windows import window_unpartition


@keras.saving.register_keras_serializable(package="paz")
class HieraPositionEmbedding(Layer):
    def __init__(self, hidden_size, background, window, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.background = tuple(background)
        self.window = window

    def build(self, input_shape):
        background = (*self.background, self.hidden_size)
        window = (self.window, self.window, self.hidden_size)
        self.background_table = self.zeros("background", background)
        self.window_table = self.zeros("window", window)

    def zeros(self, name, shape):
        return self.add_weight(name=name, shape=shape, initializer="zeros")

    def call(self, x):
        height, width = x.shape[1], x.shape[2]
        rows = bicubic_resize_matrix(self.background[0], height)
        columns = bicubic_resize_matrix(self.background[1], width)
        background = ops.einsum("hn,nmc->hmc", rows, self.background_table)
        background = ops.einsum("wm,hmc->hwc", columns, background)
        window = ops.tile(self.window_table, self.tiles(height, width))
        return x + background + window

    def tiles(self, height, width):
        return (height // self.window, width // self.window, 1)

    def get_config(self):
        config = super().get_config()
        config["hidden_size"] = self.hidden_size
        config["background"] = self.background
        config["window"] = self.window
        return config


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
        return near_cubic(distance, coefficient)
    if distance < 2:
        return far_cubic(distance, coefficient)
    return 0.0


def near_cubic(t, a):
    return ((a + 2) * t - (a + 3)) * t * t + 1


def far_cubic(t, a):
    return a * (((t - 5) * t + 8) * t - 4)


def build(images, config):
    tokens = build_patch_embedding(images, config.embed_dim)
    args = config.embed_dim, config.background, config.window_spec[0]
    tokens = HieraPositionEmbedding(*args, name="trunk_pos_embed")(tokens)
    specifications, stage_ends = build_block_specifications(config)
    outputs = []
    for specification in specifications:
        tokens = apply_block(tokens, *specification)
        if specification[0] in stage_ends:
            outputs.append(tokens)
    return outputs


def build_patch_embedding(images, hidden_size):
    padded = ZeroPadding2D(3, name="patch_embed_pad")(images)
    kwargs = dict(strides=4, padding="valid", name="patch_embed_proj")
    return Conv2D(hidden_size, 7, **kwargs)(padded)


def build_block_specifications(config):
    stages = config.stages
    global_blocks = set(config.global_blocks)
    stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
    starts = [end + 1 for end in stage_ends[:-1]]
    pool_blocks = starts[:cfg.QUERY_POOL_STAGES]
    embed_dim = config.embed_dim
    num_heads = config.num_heads
    current_stage = 1
    specifications = []
    for index in range(sum(stages)):
        dim, dim_out = embed_dim, embed_dim
        window = config.window_spec[current_stage - 1]
        if index in global_blocks:
            window = 0
        if (index - 1) in stage_ends:
            dim_out = embed_dim * cfg.DIM_MUL
            num_heads = num_heads * cfg.HEAD_MUL
            current_stage = current_stage + 1
        pool = index in pool_blocks
        name = f"trunk_block_{index}"
        spec = index, dim, dim_out, num_heads, window, pool, name
        specifications.append(spec)
        embed_dim = dim_out
    return specifications, stage_ends


def apply_block(x, index, dim, dim_out, num_heads, window, pool, name):
    shortcut = x
    normed = LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(x)
    if dim != dim_out:
        projected = Dense(dim_out, name=f"{name}_proj")(normed)
        shortcut = pool_spatial(projected) if pool else projected
    height, width = normed.shape[1], normed.shape[2]
    padded_size = (height, width)
    if window > 0:
        normed, padded_size = window_partition(normed, window)
    attended = apply_attention(normed, dim_out, num_heads, pool, f"{name}_attn")
    if pool:
        window = window // cfg.QUERY_STRIDE[0]
        height, width = shortcut.shape[1], shortcut.shape[2]
        padded_size = padded_after_pool(height, width, window)
    if window > 0:
        size = (height, width)
        attended = window_unpartition(attended, window, padded_size, size)
    tokens = ops.add(shortcut, attended)
    return apply_feedforward(tokens, dim_out, name)


def apply_attention(x, dim_out, num_heads, pool, name):
    height, width = x.shape[1], x.shape[2]
    head_dim = dim_out // num_heads
    fused = Dense(dim_out * 3, name=f"{name}_qkv")(x)
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
    return Dense(dim_out, name=f"{name}_proj")(context)


def apply_feedforward(tokens, dim_out, name):
    normed = LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(tokens)
    inner = int(dim_out * cfg.MLP_RATIO)
    names = f"{name}_mlp_fc1", f"{name}_mlp_fc2"
    forwarded = feedforward.gelu(normed, inner, dim_out, *names)
    return ops.add(tokens, forwarded)


def pool_spatial(x):
    stride = cfg.QUERY_STRIDE
    return MaxPooling2D(stride, stride, padding="valid")(x)


def padded_after_pool(height, width, window):
    pad_h = (window - height % window) % window
    pad_w = (window - width % window) % window
    return (height + pad_h, width + pad_w)
