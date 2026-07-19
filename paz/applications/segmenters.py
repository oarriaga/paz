"""Promptable image-segmentation applications built on SAM 2.

Each factory returns a callable that takes an image plus point, box, or mask
prompts and returns the selected masks and their predicted quality. The heavy
image encoder is jitted once (its input is a fixed 1024x1024), so peak memory
stays bounded; preprocessing and the light prompt decoder run eagerly because
their shapes vary per call.
"""
import numpy as np
import jax

import paz
from paz.models.foundation.sam2 import predict
from paz.models.foundation.sam2.preprocessing import preprocess_image
from paz.models.foundation.sam2.prompt_encoder import dense_positional_encoding
from paz.backend.draw import overlay_masks

BACKGROUND = (0, 0, 0)
FOREGROUND = (30, 144, 255)


def SAM(model, multimask, draw):
    encode = jax.jit(lambda pixels: model.image_encoder(pixels))
    image_pe = dense_positional_encoding(model.point_encoder)[None]
    if draw is None:
        draw = overlay_best_mask

    def encode_image(image):
        pixels = preprocess_image(image)[None]
        embedding, high_res_0, high_res_1 = encode(pixels)
        size = image.shape[:2]
        parts = (embedding, high_res_0, high_res_1, image_pe, size)
        return predict.State(model, predict.Features(*parts))

    def call(image, points=None, labels=None, box=None, mask=None):
        state = encode_image(image)
        outputs = predict.predict(state, points, labels, box, mask)
        masks, scores = predict.select(outputs[0], outputs[1], multimask)
        return np.array(masks[0]) > 0, np.array(scores[0])

    if not callable(draw):
        return call

    def call_and_draw(image, **prompts):
        masks, scores = call(image, **prompts)
        return masks, scores, draw(image, masks, scores)

    return call_and_draw


def overlay_best_mask(image, masks, scores):
    best = int(np.argmax(scores))
    colors = [BACKGROUND, FOREGROUND]
    return overlay_masks(image, masks[best].astype("int32"), colors)


def SAMHieraTiny2(multimask=True, draw=None):
    return SAM(paz.models.SAMHieraTiny2(), multimask, draw)


def SAMHieraSmall2(multimask=True, draw=None):
    return SAM(paz.models.SAMHieraSmall2(), multimask, draw)


def SAMHieraBasePlus2(multimask=True, draw=None):
    return SAM(paz.models.SAMHieraBasePlus2(), multimask, draw)


def SAMHieraLarge2(multimask=True, draw=None):
    return SAM(paz.models.SAMHieraLarge2(), multimask, draw)


def SAMHieraTiny21(multimask=True, draw=None):
    return SAM(paz.models.SAMHieraTiny21(), multimask, draw)


def SAMHieraSmall21(multimask=True, draw=None):
    return SAM(paz.models.SAMHieraSmall21(), multimask, draw)


def SAMHieraBasePlus21(multimask=True, draw=None):
    return SAM(paz.models.SAMHieraBasePlus21(), multimask, draw)


def SAMHieraLarge21(multimask=True, draw=None):
    return SAM(paz.models.SAMHieraLarge21(), multimask, draw)
