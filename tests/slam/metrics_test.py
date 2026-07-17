import numpy as np

import metrics


def rotation_z(angle):
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def to_pose(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def build_line_trajectory(num_poses):
    poses = []
    for index in range(num_poses):
        rotation = rotation_z(0.05 * index)
        poses.append(to_pose(rotation, [-float(index), 0.2, 0.0]))
    return np.stack(poses)


def test_rotation_error_identity():
    assert metrics.compute_rotation_error(np.eye(3), np.eye(3)) == 0.0


def test_rotation_error_known_angle():
    error = metrics.compute_rotation_error(np.eye(3),
                                           rotation_z(np.radians(10.0)))
    np.testing.assert_allclose(error, 10.0, atol=1e-9)


def test_translation_direction_error():
    same = metrics.compute_translation_direction_error([1.0, 0.0, 0.0],
                                                       [2.0, 0.0, 0.0])
    np.testing.assert_allclose(same, 0.0, atol=1e-6)
    orthogonal = metrics.compute_translation_direction_error(
        [1.0, 0.0, 0.0], [0.0, 3.0, 0.0])
    np.testing.assert_allclose(orthogonal, 90.0, atol=1e-9)


def test_reprojection_errors_identity_case():
    intrinsics = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0],
                           [0.0, 0.0, 1.0]])
    points3D = np.array([[0.0, 0.0, 5.0], [1.0, 0.5, 4.0]])
    points2D = np.array([[320.0, 240.0], [445.0, 302.5]])
    args = (np.eye(4), intrinsics, points3D, points2D)
    errors = metrics.compute_reprojection_errors(*args)
    np.testing.assert_allclose(errors, 0.0, atol=1e-12)


def test_ATE_recovers_rigidly_transformed_trajectory():
    poses_true = build_line_trajectory(10)
    world_transform = to_pose(rotation_z(0.5), [3.0, -2.0, 1.0])
    inverse = np.linalg.inv(world_transform)
    poses_estimated = np.stack([pose @ inverse for pose in poses_true])
    ate = metrics.compute_ATE(poses_estimated, poses_true)
    assert ate < 1e-9


def test_ATE_detects_error():
    poses_true = build_line_trajectory(10)
    poses_estimated = poses_true.copy()
    poses_estimated[5, :3, 3] += 0.5
    assert metrics.compute_ATE(poses_estimated, poses_true) > 0.01


def test_RPE_identity():
    poses = build_line_trajectory(10)
    translation_RPE, rotation_RPE = metrics.compute_RPE(poses, poses, 1)
    assert translation_RPE < 1e-12
    assert rotation_RPE < 1e-5


def test_drift_identity():
    poses = build_line_trajectory(10)
    translation_drift, rotation_drift = metrics.compute_drift(poses,
                                                              poses)
    assert translation_drift < 1e-9
    assert rotation_drift < 1e-5


def test_matrix_scale_invariant_difference():
    matrix = np.arange(9.0).reshape(3, 3) + 1.0
    zero = metrics.matrix_scale_invariant_difference(matrix,
                                                     -3.0 * matrix)
    assert zero < 1e-12
    other = np.eye(3)
    assert metrics.matrix_scale_invariant_difference(matrix, other) > 0.1


def test_inlier_precision_recall():
    estimated = np.array([True, True, False, False])
    true = np.array([True, False, True, False])
    precision, recall = metrics.compute_inlier_precision_recall(estimated,
                                                                true)
    assert precision == 0.5
    assert recall == 0.5


def test_inlier_precision_recall_perfect():
    mask = np.array([True, False, True, True])
    precision, recall = metrics.compute_inlier_precision_recall(mask,
                                                                mask)
    assert precision == 1.0
    assert recall == 1.0
