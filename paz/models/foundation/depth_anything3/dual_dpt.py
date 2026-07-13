"""DualDPT head: dense depth (+confidence) and camera rays (+confidence).

Channels-last throughout. The primary depth chain upsamples to full image
resolution; the auxiliary (ray) chain stops at the finest fusion scale
(eight times the patch grid). Depth uses exp, confidences use exp(x)+1.
Dimensions (feature_dim, out_channels, fusion_features) are size-specific.
"""
import numpy as np
from keras import ops
from keras.layers import Conv2D, Conv2DTranspose, LayerNormalization, ReLU

from paz.backend.image import resize_bilinear_align_corners

HEAD_FEATURES = 32
POS_RATIO = 0.1
OMEGA = 100.0


def build_dual_dpt(feature_maps, grid, image_shape, feature_dim, out_channels,
                   fusion_features):
    norm = LayerNormalization(epsilon=1e-5, name="head_norm")
    stages = [project_stage(feature_maps[k], norm, k, grid, image_shape,
                            feature_dim, out_channels) for k in range(4)]
    main, aux = fuse_pyramid(stages, fusion_features)
    depth, depth_confidence = depth_head(main, image_shape)
    rays, ray_confidence = ray_head(aux, image_shape)
    return depth, depth_confidence, rays, ray_confidence


def project_stage(tokens, norm, stage, grid, image_shape, feature_dim,
                  out_channels):
    grid_height, grid_width = grid
    projected = norm(tokens)
    projected = ops.reshape(projected, (-1, grid_height, grid_width, feature_dim))
    projected = Conv2D(out_channels[stage], 1, name=f"head_project_{stage}")(projected)
    projected = add_pos_embed(projected, image_shape)
    return resize_stage(projected, stage, out_channels)


def resize_stage(features, stage, out_channels):
    if stage == 0:
        return Conv2DTranspose(out_channels[0], 4, strides=4,
                               name="head_resize_0")(features)
    if stage == 1:
        return Conv2DTranspose(out_channels[1], 2, strides=2,
                               name="head_resize_1")(features)
    if stage == 2:
        return features
    return Conv2D(out_channels[3], 3, strides=2, padding="same",
                  name="head_resize_3")(features)


def fuse_pyramid(stages, fusion_features):
    lateral = [reassemble(stages[k], k, fusion_features) for k in range(4)]
    main = fuse_chain(lateral, "", fusion_features)
    aux = fuse_chain(lateral, "_aux", fusion_features)
    main = Conv2D(fusion_features // 2, 3, padding="same", name="head_out1")(main)
    aux = aux_pre_head(aux, fusion_features)
    return main, aux


def reassemble(features, stage, fusion_features):
    name = f"head_layer{stage + 1}_rn"
    return Conv2D(fusion_features, 3, padding="same", use_bias=False,
                  name=name)(features)


def fuse_chain(lateral, suffix, fusion_features):
    sizes = [ops.shape(level)[1:3] for level in lateral]
    fused = fusion_block(None, lateral[3], sizes[2], f"head_refine4{suffix}", fusion_features)
    fused = fusion_block(fused, lateral[2], sizes[1], f"head_refine3{suffix}", fusion_features)
    fused = fusion_block(fused, lateral[1], sizes[0], f"head_refine2{suffix}", fusion_features)
    double = (sizes[0][0] * 2, sizes[0][1] * 2)
    return fusion_block(fused, lateral[0], double, f"head_refine1{suffix}", fusion_features)


def fusion_block(previous, lateral, size, name, fusion_features):
    fused = lateral if previous is None else previous + residual_unit(lateral, f"{name}_unit1", fusion_features)
    fused = residual_unit(fused, f"{name}_unit2", fusion_features)
    fused = resize_bilinear_align_corners(fused, size)
    return Conv2D(fusion_features, 1, name=f"{name}_out")(fused)


def residual_unit(features, name, fusion_features):
    hidden = ReLU()(features)
    hidden = Conv2D(fusion_features, 3, padding="same", name=f"{name}_conv1")(hidden)
    hidden = ReLU()(hidden)
    hidden = Conv2D(fusion_features, 3, padding="same", name=f"{name}_conv2")(hidden)
    return features + hidden


def aux_pre_head(features, fusion_features):
    half = fusion_features // 2
    widths = (half, fusion_features, half, fusion_features, half)
    for index, width in enumerate(widths):
        features = Conv2D(width, 3, padding="same",
                          name=f"head_out1_aux_{index}")(features)
    return features


def depth_head(features, image_shape):
    upsampled = resize_bilinear_align_corners(features, image_shape[:2])
    upsampled = add_pos_embed(upsampled, image_shape)
    hidden = Conv2D(HEAD_FEATURES, 3, padding="same", name="head_out2_conv0")(upsampled)
    hidden = ReLU()(hidden)
    logits = Conv2D(2, 1, name="head_out2_conv1")(hidden)
    depth = ops.exp(logits[..., 0])
    depth_confidence = ops.exp(logits[..., 1]) + 1.0
    return depth, depth_confidence


def ray_head(features, image_shape):
    features = add_pos_embed(features, image_shape)
    hidden = Conv2D(HEAD_FEATURES, 3, padding="same", name="head_out2_aux_conv0")(features)
    hidden = LayerNormalization(epsilon=1e-5, name="head_out2_aux_ln")(hidden)
    hidden = ReLU()(hidden)
    logits = Conv2D(7, 1, name="head_out2_aux_conv1")(hidden)
    rays = logits[..., :6]
    ray_confidence = ops.exp(logits[..., 6]) + 1.0
    return rays, ray_confidence


def add_pos_embed(features, image_shape):
    _, height, width, channels = features.shape
    embedding = build_uv_position_embedding(height, width, channels, image_shape)
    return features + ops.array(embedding)


def build_uv_position_embedding(height, width, channels, image_shape):
    aspect = image_shape[1] / image_shape[0]
    grid = build_uv_grid(width, height, aspect)
    embedding = position_grid_to_embedding(grid, channels)
    return (embedding * POS_RATIO).astype("float32")


def build_uv_grid(width, height, aspect):
    diagonal = (aspect ** 2 + 1.0) ** 0.5
    span_x = aspect / diagonal
    span_y = 1.0 / diagonal
    x = np.linspace(-span_x * (width - 1) / width,
                    span_x * (width - 1) / width, width)
    y = np.linspace(-span_y * (height - 1) / height,
                    span_y * (height - 1) / height, height)
    columns, rows = np.meshgrid(x, y)
    return np.stack([columns, rows], axis=-1)


def position_grid_to_embedding(grid, channels):
    flat = grid.reshape(-1, 2)
    embed_x = sincos_embedding(channels // 2, flat[:, 0])
    embed_y = sincos_embedding(channels // 2, flat[:, 1])
    embedding = np.concatenate([embed_x, embed_y], axis=-1)
    return embedding.reshape(grid.shape[0], grid.shape[1], channels)


def sincos_embedding(dimension, positions):
    omega = np.arange(dimension // 2) / (dimension / 2.0)
    omega = 1.0 / (OMEGA ** omega)
    angles = np.outer(positions, omega)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
