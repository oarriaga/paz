import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from types import SimpleNamespace

import numpy as np

from paz.backend import poses
from paz.datasets import human36m


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


def keep_initial_translation(error, initial_translation, verbose, args):
    return SimpleNamespace(x=initial_translation)


def sum_bone_lengths(joints):
    start_joints, end_joints = human36m.human_start_joints, np.arange(1, 16)
    bones = joints[start_joints] - joints[end_joints]
    return np.linalg.norm(bones, axis=-1).sum()


def test_optimize_human_pose3D_initial_depth_uses_filtered_joints3D():
    generator = np.random.default_rng(0)
    keypoints3D = 300.0 * generator.normal(size=(1, 32, 3))
    keypoints2D = 200.0 + 100.0 * generator.uniform(size=(1, 17, 2))
    intrinsics = np.array([[1e3, 0.0, 500.0], [0.0, 1e3, 500.0], [0, 0, 1.0]])
    joints3D = keypoints3D[0, human36m.args_to_joints3D]
    joints2D = poses.filter_keypoints2D_to_h36m(keypoints2D).reshape(16, 2)
    ratio = sum_bone_lengths(joints3D) / sum_bone_lengths(joints2D)
    args = (keypoints3D, keypoints2D, keep_initial_translation, intrinsics)
    poses3D, _ = poses.optimize_human_pose3D(*args)
    depth_shift = (poses3D[0] - keypoints3D[0])[:, 2]
    assert np.allclose(depth_shift, 1e3 * ratio)
