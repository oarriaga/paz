"""SAM 2 memory encoder: fuse a frame's features with its predicted mask.

Channels-last, functional. Downsamples the high-resolution mask to the feature
grid, adds it to a projection of the image features, refines with ConvNeXt
blocks, and projects to the memory dimension. Frames where the object is
absent get the learned ``no_obj_embed_spatial`` added to every position, fed
as the ``absent`` indicator. The sinusoidal position encoding is
parameter-free and returned separately, matching ``PositionEmbeddingSine``.
"""
import math

import numpy as np
from keras import Input, Model, ops
from keras.layers import Conv2D, DepthwiseConv2D, Dense
from keras.layers import LayerNormalization, ZeroPadding2D

from paz.models.foundation.sam2.configuration import MEMORY_DIM
from paz.models.foundation.sam2.layers import ChannelScale

DOWNSAMPLE = [(0, 4), (3, 16), (6, 64), (9, 256)]


def build(name="sam2_memory_encoder"):
    pix_feat = Input((64, 64, 256), name="pix_feat")
    mask = Input((1024, 1024, 1), name="mask")
    absent = Input((1,), name="absent")
    downsampled = build_mask_downsampler(mask)
    projected = Conv2D(256, 1, name="mem_pix_proj")(pix_feat)
    fused = build_fuser(ops.add(projected, downsampled))
    features = Conv2D(MEMORY_DIM, 1, name="mem_out_proj")(fused)
    missing = absent_embedding(absent)
    return Model((pix_feat, mask, absent), features + missing, name=name)


def absent_embedding(absent):
    kwargs = dict(use_bias=False, name="no_obj_embed_spatial")
    embedded = Dense(MEMORY_DIM, **kwargs)(absent)
    return ops.reshape(embedded, (-1, 1, 1, MEMORY_DIM))


def build_mask_downsampler(mask):
    x = mask
    for index, channels in DOWNSAMPLE:
        x = ZeroPadding2D(1, name=f"mask_ds_pad_{index}")(x)
        kwargs = dict(strides=2, padding="valid", name=f"mask_ds_conv_{index}")
        x = Conv2D(channels, 3, **kwargs)(x)
        x = normalize(x, f"mask_ds_ln_{index + 1}")
        x = ops.gelu(x, approximate=False)
    return Conv2D(256, 1, name="mask_ds_final")(x)


def build_fuser(x):
    for index in range(2):
        x = apply_cxblock(x, index)
    return x


def apply_cxblock(x, index):
    name = f"fuser_{index}"
    residual = x
    x = DepthwiseConv2D(7, padding="same", name=f"{name}_dw")(x)
    x = normalize(x, f"{name}_norm")
    x = Dense(1024, name=f"{name}_pw1")(x)
    x = ops.gelu(x, approximate=False)
    x = Dense(256, name=f"{name}_pw2")(x)
    x = ChannelScale(256, name=f"{name}_gamma")(x)
    return ops.add(residual, x)


def normalize(x, name):
    return LayerNormalization(axis=-1, epsilon=1e-6, name=name)(x)


def sine_position_encoding(height, width, num_features=MEMORY_DIM):
    half = num_features // 2
    scale = 2.0 * math.pi
    rows = normalized_axis(height) * scale
    columns = normalized_axis(width) * scale
    grid_y = np.broadcast_to(rows[:, None], (height, width))
    grid_x = np.broadcast_to(columns[None, :], (height, width))
    encoded_y = interleave_sine_cosine(grid_y, half)
    encoded_x = interleave_sine_cosine(grid_x, half)
    encoding = np.concatenate([encoded_y, encoded_x], axis=-1)
    return encoding[None].astype(np.float32)


def normalized_axis(length, eps=1e-6):
    positions = np.arange(1, length + 1, dtype=np.float32)
    return positions / (positions[-1] + eps)


def interleave_sine_cosine(grid, half, temperature=10000):
    dims = np.arange(half, dtype=np.float32)
    dims = temperature ** (2.0 * (dims // 2) / half)
    angles = grid[..., None] / dims
    pairs = np.stack([np.sin(angles[..., 0::2]), np.cos(angles[..., 1::2])], 3)
    return pairs.reshape(*grid.shape, -1)
