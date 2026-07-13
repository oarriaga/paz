import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from paz.applications import MatchXFeat


def build_matcher(**kwargs):
    try:
        return MatchXFeat(top_k=1024, **kwargs)
    except Exception as error:
        pytest.skip(f"pretrained weights unavailable: {error}")


def test_match_xfeat_returns_correspondences():
    match = build_matcher(draw=False)
    image = np.random.default_rng(0).integers(0, 256, (480, 640, 3), np.uint8)
    flipped = np.ascontiguousarray(image[:, ::-1])
    points_A, points_B = match(image, flipped)
    assert points_A.shape[1] == 2
    assert len(points_A) == len(points_B)


def test_match_xfeat_draws_side_by_side():
    match = build_matcher()
    image = np.random.default_rng(1).integers(0, 256, (480, 640, 3), np.uint8)
    canvas = match(image, image)
    assert canvas.shape == (480, 1280, 3)
