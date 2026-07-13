"""Standard single-chain DPT head for monocular depth plus sky segmentation.

Reuses the DualDPT fusion pyramid. Unlike DualDPT it has one chain, no
positional embedding, and an identity input norm; it emits depth (exp) and a
sky map (relu).
"""
from keras import ops
from keras.layers import Conv2D, ReLU

from paz.backend.image import resize_bilinear_align_corners
from paz.models.foundation.depth_anything3.dual_dpt import HEAD_FEATURES
from paz.models.foundation.depth_anything3.dual_dpt import fuse_chain
from paz.models.foundation.depth_anything3.dual_dpt import reassemble
from paz.models.foundation.depth_anything3.dual_dpt import resize_stage


def build_dpt(feature_maps, grid, image_shape, feature_dim, out_channels,
              fusion_features):
    stages = [project_stage(feature_maps[k], k, grid, feature_dim, out_channels)
              for k in range(4)]
    lateral = [reassemble(stages[k], k, fusion_features) for k in range(4)]
    fused = fuse_chain(lateral, "", fusion_features)
    fused = Conv2D(fusion_features // 2, 3, padding="same", name="head_out1")(fused)
    fused = resize_bilinear_align_corners(fused, image_shape[:2])
    return apply_depth(fused), apply_sky(fused)


def project_stage(tokens, stage, grid, feature_dim, out_channels):
    grid_height, grid_width = grid
    projected = ops.reshape(tokens, (-1, grid_height, grid_width, feature_dim))
    projected = Conv2D(out_channels[stage], 1, name=f"head_project_{stage}")(projected)
    return resize_stage(projected, stage, out_channels)


def apply_depth(fused):
    hidden = Conv2D(HEAD_FEATURES, 3, padding="same", name="head_out2_conv0")(fused)
    hidden = ReLU()(hidden)
    logits = Conv2D(1, 1, name="head_out2_conv1")(hidden)
    return ops.exp(logits[..., 0])


def apply_sky(fused):
    hidden = Conv2D(HEAD_FEATURES, 3, padding="same", name="head_sky_conv0")(fused)
    hidden = ReLU()(hidden)
    logits = Conv2D(1, 1, name="head_sky_conv1")(hidden)
    return ops.relu(logits[..., 0])
