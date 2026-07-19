"""Image and prompt-coordinate preprocessing for SAM 2 image inference.

Mirrors the official ``SAM2Transforms``: scale to ``[0, 1]``, antialiased
bilinear resize to ``1024 x 1024``, then ImageNet normalization. Coordinates
arrive as ``(x, y)`` in the original image and leave in the resized ``1024``
space expected by the prompt encoder.
"""
import jax.numpy as jp

import paz
from paz.models.foundation.sam2.configuration import IMAGE_SIZE

MEAN = (0.485, 0.456, 0.406)
STDV = (0.229, 0.224, 0.225)


def preprocess_image(image, size=IMAGE_SIZE):
    image = jp.asarray(image, jp.float32) / 255.0
    image = paz.image.resize(image, (size, size), "linear", True)
    mean = jp.asarray(MEAN, jp.float32)
    stdv = jp.asarray(STDV, jp.float32)
    return (image - mean) / stdv


def transform_coords(coords, original_size, size=IMAGE_SIZE):
    height, width = original_size
    scale = jp.asarray([width, height], jp.float32)
    return jp.asarray(coords, jp.float32) / scale * size


def transform_boxes(boxes, original_size, size=IMAGE_SIZE):
    corners = jp.asarray(boxes, jp.float32).reshape(-1, 2, 2)
    return transform_coords(corners, original_size, size)
