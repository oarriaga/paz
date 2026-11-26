import numpy as np

import paz
from paz.models import XFeat
from paz.models import LighterGlue
from paz.models.feature.lightglue.model import match_pairs


def MatchXFeat(top_k=4096, threshold=0.05, min_score=0.1, draw=None):
    """Builds an image pair matcher with XFeat and LighterGlue.

    # Arguments
    - `top_k`: `top_k` passed to `XFeat`, also the `LighterGlue` capacity.
    - `threshold`: `threshold` passed to `XFeat`.
    - `min_score`: `min_score` passed to `LighterGlue`.
    - `draw`: `None` draws with `paz.draw.matches`. A callable is called as
      `draw(image_A, image_B, keypoints_A, keypoints_B)`. Any other
      value, such as `False`, disables drawing.

    # Returns
    Callable taking `image_A` and `image_B`. Without drawing it returns
    `(keypoints_A, keypoints_B)`, the matched keypoints of each image in the
    same order. With drawing it returns only the output of `draw`.
    """
    extract = XFeat("pretrained", top_k, threshold)
    match = LighterGlue("pretrained", min_score, capacity=top_k)

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
