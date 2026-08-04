"""DualDPT head: dense depth (+confidence) and camera rays (+confidence).

DPT is the dense prediction transformer head. This dual variant runs two
independent fusion chains: a primary depth chain upsampled to full image
resolution, and an auxiliary ray chain that stops at the finest fusion scale
(eight times the patch grid). Channels-last throughout. Depth uses exp,
confidences use exp(x)+1. Dimensions are size-specific.
"""
import numpy as np
from keras import ops
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import LayerNormalization, ReLU

from paz.backend.image import resize_bilinear_align_corners


def build(feature_maps, grid, image_shape, feature_dim, out_channels,
          fusion_features):
    norm = LayerNormalization(epsilon=1e-5, name="head_norm")
    stages = []
    for arg in range(4):
        args = feature_maps[arg], norm, arg, grid, image_shape, feature_dim
        stages.append(project_stage(*args, out_channels))
    main, aux = fuse_pyramid(stages, fusion_features)
    depth, depth_confidence = depth_head(main, image_shape)
    rays, ray_confidence = ray_head(aux, image_shape)
    return depth, depth_confidence, rays, ray_confidence


def project_stage(tokens, norm, stage, grid, image_shape, feature_dim,
                  out_channels):
    grid_height, grid_width = grid
    shape = -1, grid_height, grid_width, feature_dim
    reshaped = ops.reshape(norm(tokens), shape)
    projected = conv(out_channels[stage], 1, f"head_project_{stage}")(reshaped)
    projected = add_pos_embed(projected, image_shape)
    return resize_stage(projected, stage, out_channels)


def resize_stage(features, stage, out_channels):
    builders = upsample_4x, upsample_2x, keep, downsample_2x
    return builders[stage](features, out_channels[stage])


def upsample_4x(features, channels):
    layer = Conv2DTranspose(channels, 4, strides=4, name="head_resize_0")
    return layer(features)


def upsample_2x(features, channels):
    layer = Conv2DTranspose(channels, 2, strides=2, name="head_resize_1")
    return layer(features)


def keep(features, channels):
    return features


def downsample_2x(features, channels):
    kwargs = dict(strides=2, padding="same", name="head_resize_3")
    return Conv2D(channels, 3, **kwargs)(features)


def fuse_pyramid(stages, fusion_features):
    lateral = build_lateral(stages, fusion_features)
    main = fuse_chain(lateral, "", fusion_features)
    aux = fuse_chain(lateral, "_aux", fusion_features)
    main = conv(fusion_features // 2, 3, "head_out1")(main)
    aux = aux_pre_head(aux, fusion_features)
    return main, aux


def build_lateral(stages, fusion_features):
    lateral = []
    for arg in range(4):
        lateral.append(reassemble(stages[arg], arg, fusion_features))
    return lateral


def reassemble(features, stage, fusion_features):
    name = f"head_layer{stage + 1}_rn"
    return conv(fusion_features, 3, name, False)(features)


def fuse_chain(lateral, suffix, fusion_features):
    targets = fuse_targets(lateral)
    fused = None
    for step in range(4):
        name = f"head_refine{4 - step}{suffix}"
        args = fused, lateral[3 - step], targets[step], name
        fused = fusion_block(*args, fusion_features)
    return fused


def fuse_targets(lateral):
    sizes = [ops.shape(level)[1:3] for level in lateral]
    doubled = sizes[0][0] * 2, sizes[0][1] * 2
    return [sizes[2], sizes[1], sizes[0], doubled]


def fusion_block(previous, lateral, size, name, fusion_features):
    fused = fuse_lateral(previous, lateral, name, fusion_features)
    fused = residual_unit(fused, f"{name}_unit2", fusion_features)
    fused = resize_bilinear_align_corners(fused, size)
    return conv(fusion_features, 1, f"{name}_out")(fused)


def fuse_lateral(previous, lateral, name, fusion_features):
    if previous is None:
        return lateral
    return previous + residual_unit(lateral, f"{name}_unit1", fusion_features)


def residual_unit(features, name, fusion_features):
    hidden = ReLU()(features)
    hidden = conv(fusion_features, 3, f"{name}_conv1")(hidden)
    hidden = ReLU()(hidden)
    hidden = conv(fusion_features, 3, f"{name}_conv2")(hidden)
    return features + hidden


def aux_pre_head(features, fusion_features):
    half = fusion_features // 2
    widths = half, fusion_features, half, fusion_features, half
    for arg, width in enumerate(widths):
        features = conv(width, 3, f"head_out1_aux_{arg}")(features)
    return features


def depth_head(features, image_shape):
    upsampled = resize_bilinear_align_corners(features, image_shape[:2])
    upsampled = add_pos_embed(upsampled, image_shape)
    hidden = conv(32, 3, "head_out2_conv0")(upsampled)
    logits = conv(2, 1, "head_out2_conv1")(ReLU()(hidden))
    depth = ops.exp(logits[..., 0])
    return depth, ops.exp(logits[..., 1]) + 1.0


def ray_head(features, image_shape):
    features = add_pos_embed(features, image_shape)
    hidden = conv(32, 3, "head_out2_aux_conv0")(features)
    hidden = LayerNormalization(epsilon=1e-5, name="head_out2_aux_ln")(hidden)
    logits = conv(7, 1, "head_out2_aux_conv1")(ReLU()(hidden))
    return logits[..., :6], ops.exp(logits[..., 6]) + 1.0


def add_pos_embed(features, image_shape):
    batch, H, W, channels = features.shape
    embedding = build_uv_position_embedding(H, W, channels, image_shape)
    return features + ops.array(embedding)


def build_uv_position_embedding(H, W, channels, image_shape):
    aspect = image_shape[1] / image_shape[0]
    grid = build_uv_grid(W, H, aspect)
    embedding = position_grid_to_embedding(grid, channels)
    return (embedding * 0.1).astype("float32")


def build_uv_grid(W, H, aspect):
    diagonal = (aspect ** 2 + 1.0) ** 0.5
    x = build_axis(aspect / diagonal, W)
    y = build_axis(1.0 / diagonal, H)
    columns, rows = np.meshgrid(x, y)
    return np.stack([columns, rows], axis=-1)


def build_axis(span, count):
    limit = span * (count - 1) / count
    return np.linspace(-limit, limit, count)


def position_grid_to_embedding(grid, channels):
    flat = grid.reshape(-1, 2)
    embed_x = sincos_embedding(channels // 2, flat[:, 0])
    embed_y = sincos_embedding(channels // 2, flat[:, 1])
    embedding = np.concatenate([embed_x, embed_y], axis=-1)
    return embedding.reshape(grid.shape[0], grid.shape[1], channels)


def sincos_embedding(dimension, positions):
    omega = np.arange(dimension // 2) / (dimension / 2.0)
    omega = 1.0 / (100.0 ** omega)
    angles = np.outer(positions, omega)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def conv(units, kernel_size, name, use_bias=True):
    kwargs = dict(padding="same", use_bias=use_bias, name=name)
    return Conv2D(units, kernel_size, **kwargs)
