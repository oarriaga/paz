import numpy as np
import pytest

from paz.applications import MatchXFeat

pytestmark = pytest.mark.weights


def test_match_xfeat_recovers_a_horizontal_shift():
    match = MatchXFeat(top_k=1024, draw=False)
    image = np.random.default_rng(0).integers(0, 256, (480, 640, 3), np.uint8)
    points_A, points_B = match(image, np.roll(image, 40, axis=1))
    shifts = np.asarray(points_B) - np.asarray(points_A)
    assert len(points_A) > 100
    assert np.allclose(np.median(shifts, axis=0), [40, 0], atol=1)


def test_match_xfeat_draws_side_by_side():
    match = MatchXFeat(top_k=1024)
    image = np.random.default_rng(1).integers(0, 256, (480, 640, 3), np.uint8)
    canvas = match(image, image)
    assert canvas.shape == (480, 1280, 3)
