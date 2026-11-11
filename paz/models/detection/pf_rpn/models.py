"""PF-RPN: prompt-free region proposals on Grounding DINO.

The model proposes boxes for every object in an image without class names or
example images. It is Grounding DINO with a Swin-B backbone whose text encoder
is replaced by pseudo-text tokens computed from the image, and whose queries
are selected with a learned confidence.

Models return ``(scores, boxes)``: one score per query and normalized
``(cx, cy, w, h)`` boxes. Image sides must be multiples of 32.

Call the model through ``jax.jit``, as the application does. An eager call
keeps every intermediate tensor alive and needs tens of gigabytes.
"""
from keras import Input, Model
from keras.layers import Conv2D, GroupNormalization, ZeroPadding2D

from paz.models.detection.pf_rpn import decoder, encoder, prompts, swin
from paz.models.detection.rf_detr.decoder import flatten_levels, read_grids


def PFRPN(image_shape=(640, 640, 3), name="pf_rpn"):
    args = 128, (2, 2, 18, 2), (4, 8, 16, 32), 12, 6, 6, 900
    return build_model(image_shape, *args, name)


def build_model(image_shape, embed_dim, depths, num_heads, window_size,
                num_encoder_layers, num_decoder_layers, num_queries, name):
    if image_shape[0] % 32 or image_shape[1] % 32:
        raise ValueError(f"Image sides must be multiples of 32: {image_shape}")
    images = Input(image_shape)
    features = swin.build(images, embed_dim, depths, num_heads, window_size)
    features = build_neck(features, 256)
    grids, memory = read_grids(features), flatten_levels(features)
    texts = prompts.build(memory, grids)
    memory, texts = encoder.build(memory, texts, grids, num_encoder_layers)
    args = grids, num_decoder_layers, num_queries
    return Model(images, decoder.build(memory, texts, *args), name=name)


def build_neck(features, hidden_size):
    """Maps every level to ``hidden_size`` and adds one coarser level."""
    levels = []
    for level, feature in enumerate(features):
        x = Conv2D(hidden_size, 1, name=f"neck_convs_{level}_conv")(feature)
        levels.append(normalize_groups(x, f"neck_convs_{level}_gn"))
    x = ZeroPadding2D(1)(features[-1])
    x = Conv2D(hidden_size, 3, 2, name="neck_extra_convs_0_conv")(x)
    return levels + [normalize_groups(x, "neck_extra_convs_0_gn")]


def normalize_groups(x, name):
    return GroupNormalization(32, epsilon=1e-5, name=name)(x)
