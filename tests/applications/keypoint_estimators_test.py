import numpy as np
import jax.numpy as jp
import pytest
import paz
from paz.applications.keypoint_estimators import DetNetHandKeypoints


def locate_brightest_cell(images):
    cells = images[0, ::4, ::4, 0]
    uv = jp.array(jp.unravel_index(jp.argmax(cells), cells.shape))
    return jp.zeros((21, 3)), jp.tile(uv, (21, 1))


@pytest.fixture
def image():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[20:24, 40:44] = 255
    return image


@pytest.fixture
def fake_detnet(monkeypatch):
    monkeypatch.setattr(paz.models, "DetNet", lambda **_: locate_brightest_cell)


def test_left_hand_keypoints_are_x_then_y(fake_detnet, image):
    keypoints2D, _ = DetNetHandKeypoints(right_hand=False, draw=False)(image)
    assert np.all(keypoints2D == [40, 20])


def test_right_hand_keypoints_are_not_mirrored(fake_detnet, image):
    keypoints2D, _ = DetNetHandKeypoints(right_hand=True, draw=False)(image)
    assert np.all(keypoints2D == [40, 20])


def test_right_hand_equals_mirrored_left_hand(fake_detnet, image):
    left, _ = DetNetHandKeypoints(right_hand=False, draw=False)(image[:, ::-1])
    right, _ = DetNetHandKeypoints(right_hand=True, draw=False)(image)
    mirrored_left = np.stack([128 - 4 - left[:, 0], left[:, 1]], axis=1)
    assert np.all(right == mirrored_left)
