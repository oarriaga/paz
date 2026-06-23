import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from paz.backend import poses


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
