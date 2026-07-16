"""Static-image inference orchestration for SAM 2.

Encodes an image once, then predicts masks for point, box, combined, and
previous-mask prompts. Coordinates are ``(x, y)`` in the original image; boxes
are ``XYXY``. Returns masks at the original resolution, predicted quality
scores, and the low-resolution logits for a follow-up prompt.
"""
from collections import namedtuple

import jax.numpy as jp

import paz
from paz.models.foundation.sam2.preprocessing import preprocess_image
from paz.models.foundation.sam2.preprocessing import transform_coords
from paz.models.foundation.sam2.preprocessing import transform_boxes
from paz.models.foundation.sam2.prompt_encoder import dense_positional_encoding
from paz.models.foundation.sam2.prompt_encoder import MASK_INPUT

Features = namedtuple(
    "Features", "image_embed high_res_0 high_res_1 image_pe size")


def encode_image(bundle, image):
    pixels = preprocess_image(image)[None]
    embedding, high_res_0, high_res_1 = bundle.image_encoder(pixels)
    image_pe = dense_positional_encoding(bundle.point_encoder)[None]
    return Features(embedding, high_res_0, high_res_1, image_pe,
                    image.shape[:2])


def predict(bundle, features, points=None, labels=None, box=None, mask=None,
            multimask=True):
    coordinates, labels = build_prompt(points, labels, box, features.size)
    sparse = bundle.point_encoder((coordinates, labels))
    dense = build_dense(bundle, mask)
    inputs = (features.image_embed, features.high_res_0, features.high_res_1,
              sparse, dense, features.image_pe)
    masks, scores, _ = bundle.mask_decoder(inputs)
    masks, scores = select(masks, scores, multimask)
    return upscale_masks(masks, features.size), scores, masks


def build_prompt(points, labels, box, size):
    coordinates, categories = [], []
    if box is not None:
        corners = transform_boxes(box, size).reshape(-1, 2)
        coordinates.append(corners)
        categories.append(jp.array([2.0, 3.0]))
    if points is not None:
        coordinates.append(transform_coords(points, size))
        categories.append(jp.asarray(labels, jp.float32))
    coordinates.append(jp.zeros((1, 2)))
    categories.append(jp.array([-1.0]))
    coordinates = jp.concatenate(coordinates, axis=0)
    categories = jp.concatenate(categories, axis=0)
    return coordinates[None], categories[None]


def build_dense(bundle, mask):
    if mask is None:
        zeros = jp.zeros((1, MASK_INPUT, MASK_INPUT, 1), jp.float32)
        _, dense = bundle.mask_downscaling(zeros)
        return dense
    mask = jp.asarray(mask, jp.float32).reshape(1, MASK_INPUT, MASK_INPUT, 1)
    dense, _ = bundle.mask_downscaling(mask)
    return dense


def select(masks, scores, multimask):
    start = 1 if multimask else 0
    stop = masks.shape[1] if multimask else 1
    return masks[:, start:stop], scores[:, start:stop]


def upscale_masks(masks, size):
    channels_last = jp.transpose(masks[0], (1, 2, 0))
    resized = paz.image.resize(channels_last, size, "linear", False)
    return jp.transpose(resized, (2, 0, 1))[None]
