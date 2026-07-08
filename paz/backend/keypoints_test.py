from collections import namedtuple

import numpy as np
import jax.numpy as jp
import cv2

from paz.backend import keypoints
from paz.backend.poses import project_to_image

Camera = namedtuple("Camera", ["intrinsics", "distortion"])


def build_camera(size=128, focal=150.0):
    center = size / 2.0
    intrinsics = np.array([[focal, 0, center],
                           [0, focal, center],
                           [0, 0, 1.0]])
    return Camera(intrinsics, np.zeros((4, 1)))


def normalize_reference(points2D, height, width):
    image_shape = np.array([width, height])
    return 2.0 * (points2D / image_shape) - 1.0


def test_normalize_keypoints2D_matches_numpy_reference():
    points2D = np.array([[0.0, 0.0], [128.0, 64.0], [32.0, 96.0]])
    result = np.asarray(keypoints.normalize_keypoints2D(points2D, 128, 128))
    reference = normalize_reference(points2D, 128, 128)
    assert np.allclose(result, reference)


def test_normalize_denormalize_round_trip():
    points2D = jp.array([[10.0, 20.0], [50.0, 5.0], [127.0, 63.0]])
    normalized = keypoints.normalize_keypoints2D(points2D, 128, 64)
    recovered = keypoints.denormalize_keypoints2D(normalized, 128, 64)
    assert np.allclose(np.asarray(recovered), np.asarray(points2D))


def test_rotate_point2D_ninety_degrees():
    rotated = keypoints.rotate_point2D(jp.array([1.0, 0.0]), 90.0)
    assert np.allclose(np.asarray(rotated), [0.0, 1.0], atol=1e-6)


def test_flip_keypoints_left_right():
    points = jp.array([[0.0, 5.0], [32.0, 10.0]])
    flipped = np.asarray(keypoints.flip_keypoints_left_right(points, 32.0))
    assert np.allclose(flipped, [[32.0, 5.0], [0.0, 10.0]])


def test_transform_keypoint_translation():
    transform = jp.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0, 0, 1.0]])
    moved = keypoints.transform_keypoint(jp.array([4.0, 5.0]), transform)
    assert np.allclose(np.asarray(moved)[:2], [7.0, 3.0])


def test_uv_to_vu():
    flipped = keypoints.uv_to_vu(jp.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(np.asarray(flipped), [[2.0, 1.0], [4.0, 3.0]])


def test_build_cube_points3D_shape_and_center():
    cube = np.asarray(keypoints.build_cube_points3D(2.0, 4.0, 6.0))
    assert cube.shape == (8, 3)
    assert np.allclose(cube.mean(axis=0), [0.0, 0.0, 0.0])
    assert np.allclose(np.abs(cube).max(axis=0), [1.0, 2.0, 3.0])


def test_solve_PnP_recovers_known_pose():
    camera = build_camera()
    points3D = keypoints.build_cube_points3D(0.1, 0.1, 0.1)
    points3D = np.asarray(points3D, np.float64)
    rotation = cv2.Rodrigues(np.array([0.2, -0.1, 0.3]))[0]
    translation = np.array([0.02, -0.01, 0.5])
    points2D = project_to_image(rotation, translation, points3D,
                                camera.intrinsics)
    pose6D = keypoints.solve_PnP(points2D, points3D, camera)
    recovered = keypoints.rotation_vector_to_matrix(pose6D.rotation_vector)
    assert np.allclose(recovered, rotation, atol=1e-4)
    assert np.allclose(np.asarray(pose6D.translation).reshape(3),
                       translation, atol=1e-4)


def test_solve_pose_matrix_RANSAC_recovers_known_pose():
    camera = build_camera()
    grid = np.linspace(-0.05, 0.05, 5)
    points3D = np.array([[x, y, z] for x in grid for y in grid for z in grid])
    rotation = cv2.Rodrigues(np.array([0.1, 0.2, -0.15]))[0]
    translation = np.array([0.0, 0.0, 0.6])
    points2D = project_to_image(rotation, translation, points3D,
                                camera.intrinsics)
    result = keypoints.solve_pose_matrix_RANSAC(points2D, points3D, camera)
    assert result is not None
    recovered_rotation, recovered_translation = result
    assert np.allclose(recovered_rotation, rotation, atol=1e-3)
    assert np.allclose(recovered_translation, translation, atol=1e-3)


def test_solve_PnP_RANSAC_returns_none_below_minimum():
    camera = build_camera()
    points2D = np.zeros((3, 2))
    points3D = np.zeros((3, 3))
    assert keypoints.solve_PnP_RANSAC(points2D, points3D, camera) is None
