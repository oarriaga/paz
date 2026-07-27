"""Video object-tracking applications built on SAM 2.

Each factory returns a generator function: give it the video frames and the
prompts that start each object, and it yields one frame at a time with the
boolean mask of every object and, unless ``draw`` is disabled, an overlay of
them. The heavy image encoder is jitted once (its input is a fixed 1024x1024)
so peak memory stays bounded; the memory attention and prompt decoder stay
eager because their shapes grow with the memory bank.
"""
import numpy as np
import jax

import paz
from paz.models.foundation.sam2 import video
from paz.backend.draw import lincolor, overlay_masks

BACKGROUND = (0, 0, 0)


def SAM(model, draw):
    encoder = jax.jit(lambda pixels: model.image_encoder(pixels))
    bundle = model._replace(image_encoder=encoder)
    if draw is None:
        draw = overlay_objects

    def call(images, prompts):
        for frame, masks in video.track(bundle, images, prompts):
            yield frame, np.array(masks) > 0

    if not callable(draw):
        return call

    def call_and_draw(images, prompts):
        for frame, masks in call(images, prompts):
            yield frame, masks, draw(images[frame], masks)

    return call_and_draw


def overlay_objects(image, masks):
    class_map = np.zeros(masks.shape[1:], np.int32)
    for index, mask in enumerate(masks):
        class_map[mask] = index + 1
    colors = [BACKGROUND] + lincolor(len(masks))
    return overlay_masks(image, class_map, colors)


def TrackSAMHieraTiny2(draw=None):
    return SAM(paz.models.SAMHieraTiny2Video(), draw)


def TrackSAMHieraSmall2(draw=None):
    return SAM(paz.models.SAMHieraSmall2Video(), draw)


def TrackSAMHieraBasePlus2(draw=None):
    return SAM(paz.models.SAMHieraBasePlus2Video(), draw)


def TrackSAMHieraLarge2(draw=None):
    return SAM(paz.models.SAMHieraLarge2Video(), draw)


def TrackSAMHieraTiny21(draw=None):
    return SAM(paz.models.SAMHieraTiny21Video(), draw)


def TrackSAMHieraSmall21(draw=None):
    return SAM(paz.models.SAMHieraSmall21Video(), draw)


def TrackSAMHieraBasePlus21(draw=None):
    return SAM(paz.models.SAMHieraBasePlus21Video(), draw)


def TrackSAMHieraLarge21(draw=None):
    return SAM(paz.models.SAMHieraLarge21Video(), draw)
