import numpy as np

import paz
from paz.models import XFeat
from paz.models import LighterGlue
from paz.models.feature.lightglue.model import match_pairs


def MatchXFeat(top_k=4096, threshold=0.05, min_score=0.1, draw=None):
    extract = XFeat("pretrained", top_k, threshold)
    match = LighterGlue("pretrained", filter_threshold=min_score)

    def call(image_A, image_B):
        features_A = extract(image_A)
        features_B = extract(image_B)
        matches = match(features_A.keypoints, features_A.descriptors,
                        features_B.keypoints, features_B.descriptors,
                        image_size(image_A), image_size(image_B))
        pairs = match_pairs(matches)
        return (features_A.keypoints[pairs[:, 0]],
                features_B.keypoints[pairs[:, 1]])

    if draw is None:
        draw = paz.draw.matches
    return (lambda a, b: draw(a, b, *call(a, b))) if callable(draw) else call


def image_size(image):
    return np.array([image.shape[1], image.shape[0]], np.float32)
