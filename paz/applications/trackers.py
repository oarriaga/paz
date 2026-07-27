"""Video object-tracking applications built on SAM 2.

Each factory returns a generator function: give it the video frames and the
prompts that start each object, and it yields one frame at a time with the
boolean mask of every object and, unless ``draw`` is disabled, an overlay of
them. Every sub-model is jitted, and the tracker pads the memory bank to a
fixed size, so each sub-model compiles exactly once and every later frame runs
at the same cost and the same bounded memory. On a small GPU the SAM 2.1 small
backbone holds at about 0.2 s and 1.3 GB per frame.
"""
import numpy as np
import jax

import paz
from paz.models.foundation.sam2 import video
from paz.models.foundation.sam2.model import submodels
from paz.backend.draw import lincolor, overlay_masks

BACKGROUND = (0, 0, 0)


def SAM(model, draw):
    bundle = compile_submodels(model)
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


def compile_submodels(model):
    # The point encoder stays eager: the tracker reads its Fourier matrix to
    # build the dense positional encoding, and six tokens gain nothing anyway.
    compiled = {}
    for name, submodel in submodels(model):
        if name != "point_encoder":
            compiled[name] = jax.jit(submodel.__call__)
    return model._replace(**compiled)


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
