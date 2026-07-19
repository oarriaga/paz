"""Assemble the four SAM 2 image-inference sub-models into one bundle.

The image encoder runs once per image; the point encoder, mask downscaling,
and mask decoder run per prompt. Keeping them separate mirrors the official
predictor's set-image / predict split and keeps each weight set serializable.
"""
from collections import namedtuple

from paz.models.foundation.sam2 import image_encoder, prompt_encoder
from paz.models.foundation.sam2 import mask_decoder

FIELDS = "image_encoder point_encoder mask_downscaling mask_decoder config"
SAM2 = namedtuple("SAM2", FIELDS)


def build(config):
    encoder = image_encoder.build(config)
    points = prompt_encoder.build_points()
    downscaling = prompt_encoder.build_mask_downscaling()
    decoder = mask_decoder.build()
    return SAM2(encoder, points, downscaling, decoder, config)
