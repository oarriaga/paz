import numpy as np

from paz.backend import anchors


def test_translation_anchors_match_box_anchor_centers_when_not_square():
    H, W = 64, 128
    branches = [np.zeros((1, 8, 16, 3))]
    translation = anchors.build_translation_anchors((H, W), branches, 1, [1.0])
    boxes = anchors.build_anchors((H, W), branches, 1, [1.0], 4.0)
    assert translation.shape == (8 * 16, 3)
    assert np.allclose(translation[:, :2], boxes[:, :2] * np.array([W, H]))
