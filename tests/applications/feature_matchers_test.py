import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from paz.applications import MatchXFeat

pytestmark = pytest.mark.weights


def test_match_xfeat_returns_correspondences():
    match = MatchXFeat(top_k=1024, draw=False)
    image = np.random.default_rng(0).integers(0, 256, (480, 640, 3), np.uint8)
    flipped = np.ascontiguousarray(image[:, ::-1])
    points_A, points_B = match(image, flipped)
    assert points_A.shape[1] == 2
    assert len(points_A) == len(points_B)


def test_match_xfeat_draws_side_by_side():
    match = MatchXFeat(top_k=1024)
    image = np.random.default_rng(1).integers(0, 256, (480, 640, 3), np.uint8)
    canvas = match(image, image)
    assert canvas.shape == (480, 1280, 3)
