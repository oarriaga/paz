# Developer-only OpenCV and SciPy oracles for the SLAM test suite.
# Production paz code must never import this module.
from collections import namedtuple

import cv2
import numpy as np
import scipy
import scipy.optimize
import scipy.sparse

ReferenceVersions = namedtuple("ReferenceVersions",
                               ["cv2", "scipy", "numpy"])

BundleResult = namedtuple(
    "BundleResult", ["poses", "points3D", "initial_rmse", "final_rmse"])


def get_reference_versions():
    versions = (cv2.__version__, scipy.__version__, np.__version__)
    return ReferenceVersions(*versions)


def estimate_fundamental_reference(points_A, points_B):
    points_A = to_float64(points_A)
    points_B = to_float64(points_B)
    fundamental, _ = cv2.findFundamentalMat(points_A, points_B,
                                            cv2.FM_8POINT)
    return fundamental


def estimate_fundamental_ransac_reference(points_A, points_B, threshold):
    points_A = to_float64(points_A)
    points_B = to_float64(points_B)
    args = (points_A, points_B, cv2.FM_RANSAC, threshold, 0.999, 10000)
    fundamental, mask = cv2.findFundamentalMat(*args)
    return fundamental, mask.ravel() > 0


def recover_pose_reference(
    essential, points_A, points_B, intrinsics_A, intrinsics_B
):
    normalized_A = normalize_pixels(intrinsics_A, points_A)
    normalized_B = normalize_pixels(intrinsics_B, points_B)
    essential = to_float64(essential)
    args = (essential, normalized_A, normalized_B, np.eye(3))
    count, rotation, translation, mask = cv2.recoverPose(*args)
    return rotation, translation.ravel(), mask.ravel() > 0


def triangulate_reference(
    intrinsics_A, pose_A, intrinsics_B, pose_B, points_A, points_B
):
    projection_A = to_float64(intrinsics_A) @ to_float64(pose_A)[:3]
    projection_B = to_float64(intrinsics_B) @ to_float64(pose_B)[:3]
    points_A = np.ascontiguousarray(to_float64(points_A).T)
    points_B = np.ascontiguousarray(to_float64(points_B).T)
    args = (projection_A, projection_B, points_A, points_B)
    points4D = cv2.triangulatePoints(*args)
    return (points4D[:3] / points4D[3]).T


def solve_pnp_reference(intrinsics, points3D, points2D):
    intrinsics = to_float64(intrinsics)
    points3D = np.ascontiguousarray(to_float64(points3D))
    points2D = np.ascontiguousarray(to_float64(points2D))
    args = (points3D, points2D, intrinsics, None)
    kwargs = dict(reprojectionError=2.0, iterationsCount=500,
                  confidence=0.9999)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(*args, **kwargs)
    inlier_mask = np.zeros(len(points3D), dtype=bool)
    inlier_mask[inliers.ravel()] = True
    refine = (points3D[inlier_mask], points2D[inlier_mask], intrinsics,
              None, rvec, tvec, True)
    success, rvec, tvec = cv2.solvePnP(*refine)
    return to_pose_matrix(rvec, tvec), inlier_mask


def refine_pose_reference(intrinsics, points3D, points2D, initial_pose):
    intrinsics = to_float64(intrinsics)
    points3D = to_float64(points3D)
    points2D = to_float64(points2D)
    parameters = pack_pose(to_float64(initial_pose))

    def compute_residuals(parameters):
        pose = unpack_pose(parameters)
        return (project(intrinsics, pose, points3D) - points2D).ravel()

    args = (compute_residuals, parameters)
    result = scipy.optimize.least_squares(*args, method="lm")
    return unpack_pose(result.x)


def bundle_adjust_reference(
    intrinsics, poses, points3D, observations, visibility
):
    intrinsics = to_float64(intrinsics)
    poses = to_float64(poses)
    points3D = to_float64(points3D)
    observations = to_float64(observations)
    visibility = np.asarray(visibility, dtype=bool)
    pose_indices, point_indices = np.nonzero(visibility)
    targets = observations[pose_indices, point_indices]
    num_poses = len(poses)

    def compute_residuals(parameters):
        args = (parameters, num_poses, intrinsics, targets,
                pose_indices, point_indices)
        return compute_bundle_residuals(*args)

    parameters = pack_bundle(poses, points3D)
    initial_rmse = compute_rmse(compute_residuals(parameters))
    args = (pose_indices, point_indices, num_poses, len(points3D))
    sparsity = build_bundle_sparsity(*args)
    kwargs = dict(jac_sparsity=sparsity, method="trf", x_scale="jac",
                  ftol=1e-8, xtol=1e-8)
    args = (compute_residuals, parameters)
    result = scipy.optimize.least_squares(*args, **kwargs)
    final_rmse = compute_rmse(result.fun)
    refined_poses, refined_points3D = unpack_bundle(result.x, num_poses)
    outcome = (refined_poses, refined_points3D, initial_rmse, final_rmse)
    return BundleResult(*outcome)


def compute_bundle_residuals(
    parameters, num_poses, intrinsics, targets, pose_indices, point_indices
):
    poses, points3D = unpack_bundle(parameters, num_poses)
    residuals = np.zeros((len(targets), 2))
    for pose_index in range(num_poses):
        selected = pose_indices == pose_index
        selected_points = points3D[point_indices[selected]]
        projected = project(intrinsics, poses[pose_index], selected_points)
        residuals[selected] = projected - targets[selected]
    return residuals.ravel()


def build_bundle_sparsity(
    pose_indices, point_indices, num_poses, num_points
):
    num_observations = len(pose_indices)
    shape = (2 * num_observations, 6 * num_poses + 3 * num_points)
    sparsity = scipy.sparse.lil_matrix(shape, dtype=int)
    rows = np.arange(num_observations)
    for axis in (0, 1):
        for offset in range(6):
            sparsity[2 * rows + axis, 6 * pose_indices + offset] = 1
        for offset in range(3):
            columns = 6 * num_poses + 3 * point_indices + offset
            sparsity[2 * rows + axis, columns] = 1
    return sparsity


def pack_bundle(poses, points3D):
    pose_blocks = [pack_pose(pose) for pose in poses]
    return np.concatenate(pose_blocks + [points3D.ravel()])


def unpack_bundle(parameters, num_poses):
    pose_parameters = parameters[:6 * num_poses].reshape(num_poses, 6)
    poses = np.stack([unpack_pose(block) for block in pose_parameters])
    points3D = parameters[6 * num_poses:].reshape(-1, 3)
    return poses, points3D


def compute_rmse(residuals):
    errors = residuals.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))


def pack_pose(pose):
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    return np.concatenate([rvec.ravel(), pose[:3, 3]])


def unpack_pose(parameters):
    return to_pose_matrix(parameters[:3], parameters[3:6])


def to_pose_matrix(rvec, tvec):
    rotation, _ = cv2.Rodrigues(to_float64(rvec))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = to_float64(tvec).ravel()
    return pose


def project(intrinsics, pose, points3D):
    camera_matrix = intrinsics @ pose[:3]
    ones = np.ones((len(points3D), 1))
    homogeneous = np.concatenate([points3D, ones], axis=1)
    projected = homogeneous @ camera_matrix.T
    return projected[:, :2] / projected[:, 2:3]


def normalize_pixels(intrinsics, points2D):
    inverse = np.linalg.inv(to_float64(intrinsics))
    points2D = to_float64(points2D)
    ones = np.ones((len(points2D), 1))
    homogeneous = np.concatenate([points2D, ones], axis=1)
    normalized = homogeneous @ inverse.T
    return np.ascontiguousarray(normalized[:, :2])


def to_float64(array):
    return np.asarray(array, dtype=np.float64)
