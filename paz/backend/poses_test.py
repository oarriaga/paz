import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from collections import namedtuple

import numpy as np
import cv2

from paz.backend import poses

Camera = namedtuple("Camera", ["intrinsics", "distortion"])


def build_camera(size=128, focal=150.0):
    center = size / 2.0
    intrinsics = np.array([[focal, 0, center],
                           [0, focal, center],
                           [0, 0, 1.0]])
    return Camera(intrinsics, np.zeros((4, 1)))


def test_rotation_matrix_to_axis_angle_identity():
    identity = np.eye(3).reshape(1, 9)
    axis_angle = poses.rotation_matrix_to_axis_angle(identity)
    assert axis_angle.shape == (1, 5)
    assert np.allclose(axis_angle[0, :3], 0.0)


def test_rotation_matrix_to_axis_angle_z_rotation():
    angle = np.pi / 2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1.0]]).reshape(1, 9)
    axis_angle = poses.rotation_matrix_to_axis_angle(rotation)
    assert np.allclose(axis_angle[0, :3], [0, 0, 0.5], atol=1e-5)


def test_match_poses_shapes_and_flag():
    prior_boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.05, 0.05]])
    boxes = np.array([[0.4, 0.4, 0.6, 0.6, 1.0]])
    pose = np.array([[0.1, 0.2, 0.3, 0.0, 1.0, 0.0, 10.0, 20.0, 500.0]])
    matched = poses.match_poses(boxes, pose, prior_boxes, iou_threshold=0.5)
    assert matched.shape == (2, 10)
    assert matched[0, -1] == 1.0
    assert matched[1, -1] == 0.0


def test_solve_PnP_recovers_known_pose():
    camera = build_camera()
    grid = np.linspace(-0.05, 0.05, 3)
    points3D = np.array([[x, y, z] for x in grid for y in grid for z in grid])
    rotation = cv2.Rodrigues(np.array([0.2, -0.1, 0.3]))[0]
    translation = np.array([0.02, -0.01, 0.5])
    points2D = poses.project_to_image(rotation, translation, points3D,
                                      camera.intrinsics)
    pose6D = poses.solve_PnP(points2D, points3D, camera)
    recovered = poses.rotation_vector_to_matrix(pose6D.rotation_vector)
    assert np.allclose(recovered, rotation, atol=1e-4)
    assert np.allclose(np.asarray(pose6D.translation).reshape(3),
                       translation, atol=1e-4)


def test_solve_pose_matrix_RANSAC_recovers_known_pose():
    camera = build_camera()
    grid = np.linspace(-0.05, 0.05, 5)
    points3D = np.array([[x, y, z] for x in grid for y in grid for z in grid])
    rotation = cv2.Rodrigues(np.array([0.1, 0.2, -0.15]))[0]
    translation = np.array([0.0, 0.0, 0.6])
    points2D = poses.project_to_image(rotation, translation, points3D,
                                      camera.intrinsics)
    result = poses.solve_pose_matrix_RANSAC(points2D, points3D, camera)
    assert result is not None
    recovered_rotation, recovered_translation = result
    assert np.allclose(recovered_rotation, rotation, atol=1e-3)
    assert np.allclose(recovered_translation, translation, atol=1e-3)


def test_solve_PnP_RANSAC_returns_none_below_minimum():
    camera = build_camera()
    points2D, points3D = np.zeros((3, 2)), np.zeros((3, 3))
    assert poses.solve_PnP_RANSAC(points2D, points3D, camera) is None


def test_project_points3D_matches_project_to_image():
    camera = build_camera()
    points3D = np.array([[0.01, 0.0, 0.5], [-0.02, 0.03, 0.6]])
    rotation = cv2.Rodrigues(np.array([0.1, 0.0, 0.0]))[0]
    translation = np.array([0.0, 0.0, 0.5])
    pose6D = poses.Pose6D(cv2.Rodrigues(rotation)[0], translation)
    cv2_points = np.asarray(poses.project_points3D(points3D, pose6D, camera))
    analytic = poses.project_to_image(rotation, translation, points3D,
                                      camera.intrinsics)
    assert np.allclose(cv2_points, analytic, atol=1e-3)
