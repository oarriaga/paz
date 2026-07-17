import numpy as np


def compute_rotation_error(R_a, R_b):
    relative = np.asarray(R_a).T @ np.asarray(R_b)
    cosine = (np.trace(relative) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def compute_translation_direction_error(t_a, t_b):
    cosine = np.dot(normalize(t_a), normalize(t_b))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def compute_reprojection_errors(pose, intrinsics, points3D, points2D):
    points3D = np.asarray(points3D, dtype=np.float64)
    points2D = np.asarray(points2D, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics) @ np.asarray(pose)[:3]
    ones = np.ones((len(points3D), 1))
    homogeneous = np.concatenate([points3D, ones], axis=1)
    projected = homogeneous @ camera_matrix.T
    projected = projected[:, :2] / projected[:, 2:3]
    return np.linalg.norm(projected - points2D, axis=1)


def compute_ATE(poses_estimated, poses_true):
    estimated = compute_camera_centers(poses_estimated)
    true = compute_camera_centers(poses_true)
    rotation, translation = align_rigid(estimated, true)
    aligned = estimated @ rotation.T + translation
    errors = np.linalg.norm(aligned - true, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def compute_RPE(poses_estimated, poses_true, delta):
    estimated = np.asarray(poses_estimated, dtype=np.float64)
    true = np.asarray(poses_true, dtype=np.float64)
    translation_errors, rotation_errors = [], []
    for index in range(len(estimated) - delta):
        error = compute_relative_error(estimated, true, index, delta)
        translation_errors.append(np.linalg.norm(error[:3, 3]))
        rotation_errors.append(compute_rotation_error(np.eye(3),
                                                      error[:3, :3]))
    translation_RPE = float(np.sqrt(np.mean(np.square(translation_errors))))
    rotation_RPE = float(np.sqrt(np.mean(np.square(rotation_errors))))
    return translation_RPE, rotation_RPE


def compute_drift(poses_estimated, poses_true):
    length = compute_trajectory_length(poses_true)
    estimated = np.asarray(poses_estimated, dtype=np.float64)
    true = np.asarray(poses_true, dtype=np.float64)
    last = len(true) - 1
    error = compute_relative_error(estimated, true, 0, last)
    translation_drift = 100.0 * np.linalg.norm(error[:3, 3]) / length
    rotation_drift = compute_rotation_error(np.eye(3),
                                            error[:3, :3]) / length
    return float(translation_drift), float(rotation_drift)


def matrix_scale_invariant_difference(F_a, F_b):
    unit_a = np.asarray(F_a) / np.linalg.norm(F_a)
    unit_b = np.asarray(F_b) / np.linalg.norm(F_b)
    difference = np.linalg.norm(unit_a - unit_b)
    flipped = np.linalg.norm(unit_a + unit_b)
    return float(min(difference, flipped))


def compute_inlier_precision_recall(estimated_mask, true_mask):
    estimated = np.asarray(estimated_mask).astype(bool)
    true = np.asarray(true_mask).astype(bool)
    true_positives = np.sum(estimated & true)
    precision = true_positives / max(np.sum(estimated), 1)
    recall = true_positives / max(np.sum(true), 1)
    return float(precision), float(recall)


def compute_relative_error(estimated, true, index, delta):
    relative_estimated = compute_relative_motion(estimated, index, delta)
    relative_true = compute_relative_motion(true, index, delta)
    return np.linalg.inv(relative_true) @ relative_estimated


def compute_relative_motion(poses, index, delta):
    return poses[index + delta] @ np.linalg.inv(poses[index])


def compute_trajectory_length(poses):
    centers = compute_camera_centers(poses)
    steps = np.diff(centers, axis=0)
    return float(np.sum(np.linalg.norm(steps, axis=1)))


def compute_camera_centers(poses):
    poses = np.asarray(poses, dtype=np.float64)
    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3]
    return -np.einsum("nij,ni->nj", rotations, translations)


def align_rigid(source, target):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (target - target_mean).T @ (source - source_mean)
    U, _, Vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(U @ Vt))
    rotation = U @ np.diag([1.0, 1.0, sign]) @ Vt
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    return vector / np.linalg.norm(vector)
