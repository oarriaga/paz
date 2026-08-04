"""SAM 2 image encoder: Hiera trunk, FPN neck, and SAM decoder projections.

Outputs the 64x64 image embedding (with ``no_mem_embed`` added, as the image
predictor does), the two high-resolution features that the mask decoder
fuses, and the plain trunk features before ``no_mem_embed``, which the video
memory path encodes and conditions on. The 1x1 ``conv_s0``/``conv_s1``
projections belong to the mask decoder in the checkpoint but run here once per
image, matching ``forward_image``.
"""
from keras import Input, Model
from keras.layers import Conv2D

from paz.models.foundation.sam2 import hiera, neck
from paz.models.foundation.sam2.configuration import IMAGE_SIZE
from paz.models.foundation.sam2.layers import ChannelBias


def build(config, name="sam2_image_encoder"):
    images = Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="pixels")
    trunk = hiera.build(images, config)
    features = neck.build(trunk)[:3]
    high_res_0 = Conv2D(32, 1, name="sam_mask_decoder_conv_s0")(features[0])
    high_res_1 = Conv2D(64, 1, name="sam_mask_decoder_conv_s1")(features[1])
    embedding = ChannelBias(256, name="no_mem_embed")(features[2])
    outputs = (embedding, high_res_0, high_res_1, features[2])
    return Model(images, outputs, name=name)
