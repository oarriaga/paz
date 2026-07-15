"""Standard single-chain DPT head for monocular depth plus sky segmentation.

DPT is the dense prediction transformer head. Unlike DualDPT it has one
fusion chain, no positional embedding, and an identity input norm; it emits
depth (exp) and a sky map (relu). The pyramid helpers are reused from DualDPT.
"""
from keras import ops
from keras.layers import Conv2D, ReLU

from paz.backend.image import resize_bilinear_align_corners
from paz.models.foundation.depth_anything3 import dual_dpt


def build(feature_maps, grid, image_shape, feature_dim, out_channels,
          fusion_features):
    stages = []
    for arg in range(4):
        args = feature_maps[arg], arg, grid, feature_dim, out_channels
        stages.append(project_stage(*args))
    lateral = dual_dpt.build_lateral(stages, fusion_features)
    fused = dual_dpt.fuse_chain(lateral, "", fusion_features)
    fused = conv(fusion_features // 2, 3, "head_out1")(fused)
    fused = resize_bilinear_align_corners(fused, image_shape[:2])
    return apply_depth(fused), apply_sky(fused)


def project_stage(tokens, stage, grid, feature_dim, out_channels):
    grid_height, grid_width = grid
    shape = -1, grid_height, grid_width, feature_dim
    projected = ops.reshape(tokens, shape)
    projected = conv(out_channels[stage], 1, f"head_project_{stage}")(projected)
    return dual_dpt.resize_stage(projected, stage, out_channels)


def apply_depth(fused):
    hidden = conv(32, 3, "head_out2_conv0")(fused)
    hidden = ReLU()(hidden)
    logits = conv(1, 1, "head_out2_conv1")(hidden)
    return ops.exp(logits[..., 0])


def apply_sky(fused):
    hidden = conv(32, 3, "head_sky_conv0")(fused)
    hidden = ReLU()(hidden)
    logits = conv(1, 1, "head_sky_conv1")(hidden)
    return ops.relu(logits[..., 0])


def conv(units, kernel_size, name):
    return Conv2D(units, kernel_size, padding="same", name=name)
