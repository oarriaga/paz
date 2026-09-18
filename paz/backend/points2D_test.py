import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import jax.numpy as jp

import paz
from paz.backend import points2D


def normalize_reference(points, height, width):
    image_shape = np.array([width, height])
    return 2.0 * (points / image_shape) - 1.0


def hot_pixel_location(image):
    row, col = np.unravel_index(np.argmax(image[..., 0]), image.shape[:2])
    return col, row


def test_normalize_keypoints2D_matches_numpy_reference():
    points = np.array([[0.0, 0.0], [128.0, 64.0], [32.0, 96.0]])
    result = np.asarray(points2D.normalize_keypoints2D(points, 128, 128))
    assert np.allclose(result, normalize_reference(points, 128, 128))


def test_normalize_denormalize_round_trip():
    points = jp.array([[10.0, 20.0], [50.0, 5.0], [127.0, 63.0]])
    normalized = points2D.normalize_keypoints2D(points, 128, 64)
    recovered = points2D.denormalize_keypoints2D(normalized, 128, 64)
    assert np.allclose(np.asarray(recovered), np.asarray(points))


def test_rotate_point2D_ninety_degrees():
    rotated = points2D.rotate_point2D(jp.array([1.0, 0.0]), 90.0)
    assert np.allclose(np.asarray(rotated), [0.0, 1.0], atol=1e-6)


def test_flip_keypoints_left_right():
    points = jp.array([[0.0, 5.0], [32.0, 10.0]])
    flipped = np.asarray(points2D.flip_keypoints_left_right(points, 32.0))
    assert np.allclose(flipped, [[32.0, 5.0], [0.0, 10.0]])


def test_uv_to_vu():
    flipped = points2D.uv_to_vu(jp.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(np.asarray(flipped), [[2.0, 1.0], [4.0, 3.0]])


def test_rotate_keypoints2D_tracks_image_rotation():
    image = np.zeros((31, 31, 3), "float32")
    image[5, 20] = 1.0
    rotated = paz.image.rotate(jp.asarray(image), 0.5)
    col, row = hot_pixel_location(np.asarray(rotated))
    center = jp.array([(31 - 1) / 2.0, (31 - 1) / 2.0])
    keypoint = jp.array([[20.0, 5.0]])  # (x=col, y=row)
    moved = np.asarray(points2D.rotate_keypoints2D(keypoint, 0.5, center))[0]
    assert abs(moved[0] - col) <= 1.5 and abs(moved[1] - row) <= 1.5


def test_translate_keypoints_tracks_image_translation():
    image = np.zeros((31, 31, 3), "float32")
    image[10, 8] = 1.0
    translation = jp.array([4.0, -3.0])  # (x, y) shift
    translated = paz.image.translate(jp.asarray(image), translation)
    col, row = hot_pixel_location(np.asarray(translated))
    keypoint = jp.array([[8.0, 10.0]])
    moved = np.asarray(points2D.translate_keypoints(keypoint, translation))[0]
    assert (moved[0], moved[1]) == (col, row)
