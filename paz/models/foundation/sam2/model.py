"""Assemble the SAM 2 sub-models into an image or a video bundle.

The image encoder runs once per image; the point encoder, mask downscaling,
and mask decoder run per prompt. Keeping them separate mirrors the official
predictor's set-image / predict split and keeps each weight set serializable.
The video bundle adds the memory encoder and memory attention, the object
pointer projections, and the mask-prompt downsampling conv; it also answers to
every image field, so the image predictor accepts it unchanged.
"""
from collections import namedtuple

from paz.models.foundation.sam2 import image_encoder, prompt_encoder
from paz.models.foundation.sam2 import mask_decoder, pointer
from paz.models.foundation.sam2 import memory_encoder, memory_attention

IMAGE = "image_encoder point_encoder mask_downscaling mask_decoder"
VIDEO = "mask_downsample memory_encoder memory_attention pointer pointer_time"
SAM2 = namedtuple("SAM2", f"{IMAGE} config")
SAM2Video = namedtuple("SAM2Video", f"{IMAGE} {VIDEO} config")


def build(config):
    encoder = image_encoder.build(config)
    points = prompt_encoder.build_points()
    downscaling = prompt_encoder.build_mask_downscaling()
    decoder = mask_decoder.build()
    return SAM2(encoder, points, downscaling, decoder, config)


def build_video(config):
    image = build(config)
    downsample = prompt_encoder.build_mask_downsample()
    memory = memory_encoder.build(), memory_attention.build()
    pointers = pointer.build(), pointer.build_time()
    return SAM2Video(*image[:4], downsample, *memory, *pointers, config)


def submodels(bundle):
    names = [name for name in bundle._fields if name != "config"]
    return [(name, getattr(bundle, name)) for name in names]
