"""Fuses the tapped DINOv2 maps into one detection feature map.

The tapped maps all share the backbone's patch grid, so fusing is a channel
concatenation followed by a CSP block: split, run bottlenecks on the second
half, concatenate every intermediate, project back. This is the ``P4`` stage
of LW-DETR's multi-scale projector, the only stage the published RF-DETR
detectors use.
"""
from keras import ops
from keras.layers import Conv2D, LayerNormalization


def build(features, out_channels, num_blocks, name):
    fused = ops.concatenate(features, axis=-1)
    refined = build_cross_stage(fused, out_channels, num_blocks, name)
    return LayerNormalization(epsilon=1e-6, name=f"{name}_norm")(refined)


def build_cross_stage(x, out_channels, num_blocks, name):
    inner_channels = out_channels // 2
    branch = build_convolution(x, 2 * inner_channels, 1, f"{name}_cv1")
    parts = list(ops.split(branch, 2, axis=-1))
    for index in range(num_blocks):
        block = f"{name}_m_{index}"
        parts.append(build_bottleneck(parts[-1], inner_channels, block))
    merged = ops.concatenate(parts, axis=-1)
    return build_convolution(merged, out_channels, 1, f"{name}_cv2")


def build_bottleneck(x, channels, name):
    hidden = build_convolution(x, channels, 3, f"{name}_cv1")
    return build_convolution(hidden, channels, 3, f"{name}_cv2")


def build_convolution(x, filters, kernel_size, name):
    kwargs = dict(padding="same", use_bias=False, name=f"{name}_conv")
    convolved = Conv2D(filters, kernel_size, **kwargs)(x)
    normalize = LayerNormalization(epsilon=1e-6, name=f"{name}_norm")
    return ops.silu(normalize(convolved))
