import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import cv2
import jax
import jax.numpy as jp
import numpy as np
from scipy.optimize import least_squares

import paz
from paz.backend import pnp


def build_intrinsics():
    return jp.array([[500.0, 0.0, 320.0],
                     [0.0, 500.0, 240.0],
                     [0.0, 0.0, 1.0]])


def build_scene(seed=0, num_points=120):
    rng = np.random.default_rng(seed)
    points3D = jp.array(rng.uniform(-2.0, 2.0, (num_points, 3)))
    points3D = points3D + jp.array([0.0, 0.0, 6.0])
    rotation = paz.SO3.rotation_y(0.2) @ paz.SO3.rotation_x(-0.1)
    translation = jp.array([0.3, -0.2, 0.5])
    pose = paz.pinhole.to_affine_matrix(rotation, translation)
    return points3D, pose


def build_noisy_scene(seed=1, num_points=100, noise=0.5, num_outliers=20):
    intrinsics = build_intrinsics()
    points3D, pose = build_scene(seed, num_points)
    points2D = project(intrinsics, pose, points3D)
    rng = np.random.default_rng(seed + 100)
    points2D = points2D + jp.array(rng.normal(0.0, noise, (num_points, 2)))
    shifts = jp.array(rng.uniform(30.0, 80.0, (num_outliers, 2)))
    points2D = points2D.at[:num_outliers].add(shifts)
    return intrinsics, points3D, points2D, pose


def project(intrinsics, pose, points3D):
    camera_matrix = paz.pinhole.make_camera_matrix(intrinsics, pose)
    pixels = (camera_matrix @ paz.algebra.add_ones(points3D).T).T
    return pixels[:, :2] / pixels[:, 2:3]


def perturb_pose(pose, angle_degrees, translation_shift):
    angular = jp.deg2rad(angle_degrees) * jp.array([0.6, -0.8, 0.0])
    linear = translation_shift * jp.array([0.0, 0.6, 0.8])
    return paz.SE3.retract(pose, jp.concatenate([angular, linear]))


def compute_rotation_degrees(pose_A, pose_B):
    cosine = (jp.trace(pose_A[:3, :3] @ pose_B[:3, :3].T) - 1.0) / 2.0
    return jp.rad2deg(jp.arccos(jp.clip(cosine, -1.0, 1.0)))


def compute_translation_error(pose_A, pose_B):
    return jp.linalg.norm(pose_A[:3, 3] - pose_B[:3, 3])


def compute_RMSE(pose, points3D, points2D, intrinsics, valid_mask):
    args = (pose, points3D, points2D, intrinsics, valid_mask)
    residuals = pnp.compute_reprojection_residuals(*args)
    squared_norms = jp.sum(residuals ** 2, axis=1)
    weights = valid_mask.astype(squared_norms.dtype)
    return jp.sqrt(jp.sum(weights * squared_norms) / jp.sum(weights))


def refine_with_scipy(initial_pose, points3D, points2D, intrinsics):
    points3D = np.array(points3D, np.float64)
    points2D = np.array(points2D, np.float64)
    K = np.array(intrinsics, np.float64)
    rvec_0, _ = cv2.Rodrigues(np.array(initial_pose[:3, :3], np.float64))
    tvec_0 = np.array(initial_pose[:3, 3], np.float64)
    x_0 = np.concatenate([rvec_0[:, 0], tvec_0])

    def residuals(x):
        rotation, _ = cv2.Rodrigues(x[:3])
        camera_points = points3D @ rotation.T + x[3:]
        pixels = camera_points @ K.T
        projected = pixels[:, :2] / pixels[:, 2:3]
        return (projected - points2D).ravel()

    solution = least_squares(residuals, x_0)
    return np.sqrt(2.0 * np.mean(solution.fun ** 2))


def solve_with_cv2(points3D, points2D, intrinsics, flags):
    cv2_args = (np.array(points3D, np.float64),
                np.array(points2D, np.float64),
                np.array(intrinsics, np.float64), None)
    _, rvec, tvec = cv2.solvePnP(*cv2_args, flags=flags)
    rotation, _ = cv2.Rodrigues(rvec)
    return jp.array(paz.pinhole.to_affine_matrix(
        jp.array(rotation), jp.array(tvec[:, 0])))


def test_solve_DLT_noise_free():
    intrinsics = build_intrinsics()
    points3D, pose = build_scene()
    points2D = project(intrinsics, pose, points3D)
    valid_mask = jp.ones(len(points2D), dtype=bool)
    estimate = pnp.solve_DLT(points3D, points2D, intrinsics, valid_mask)
    assert bool(estimate.valid)
    assert compute_rotation_degrees(estimate.pose, pose) < 0.05
    assert compute_translation_error(estimate.pose, pose) < 1e-4
    RMSE_args = (estimate.pose, points3D, points2D, intrinsics, valid_mask)
    assert compute_RMSE(*RMSE_args) < 1e-3


def test_solve_DLT_ignores_masked_points():
    intrinsics = build_intrinsics()
    points3D, pose = build_scene()
    points2D = project(intrinsics, pose, points3D)
    points2D = points2D.at[:30].add(500.0)
    valid_mask = jp.arange(len(points2D)) >= 30
    estimate = pnp.solve_DLT(points3D, points2D, intrinsics, valid_mask)
    assert bool(estimate.valid)
    assert compute_rotation_degrees(estimate.pose, pose) < 0.05
    assert compute_translation_error(estimate.pose, pose) < 1e-4


def test_solve_DLT_matches_cv2():
    intrinsics = build_intrinsics()
    points3D, pose = build_scene()
    points2D = project(intrinsics, pose, points3D)
    valid_mask = jp.ones(len(points2D), dtype=bool)
    estimate = pnp.solve_DLT(points3D, points2D, intrinsics, valid_mask)
    pose_cv2 = solve_with_cv2(
        points3D, points2D, intrinsics, cv2.SOLVEPNP_ITERATIVE)
    assert compute_rotation_degrees(estimate.pose, pose_cv2) < 0.05
    assert compute_translation_error(estimate.pose, pose_cv2) < 1e-3


def test_RANSAC_with_noise_and_outliers():
    scene = build_noisy_scene()
    intrinsics, points3D, points2D, pose = scene
    num_outliers, num_points = 20, len(points2D)
    valid_mask = jp.ones(num_points, dtype=bool)
    key = jax.random.PRNGKey(0)
    RANSAC_args = (key, points3D, points2D, intrinsics, valid_mask, 200, 2.5)
    estimate = pnp.estimate_pose_RANSAC(*RANSAC_args)
    assert bool(estimate.valid)
    assert compute_rotation_degrees(estimate.pose, pose) < 0.5
    assert compute_translation_error(estimate.pose, pose) < 0.06
    inliers = np.array(estimate.inliers)
    num_true_positives = inliers[num_outliers:].sum()
    precision = num_true_positives / max(inliers.sum(), 1)
    recall = num_true_positives / (num_points - num_outliers)
    assert precision >= 0.97
    assert recall >= 0.9
    assert int(estimate.num_inliers) == inliers.sum()


def test_RANSAC_matches_cv2():
    scene = build_noisy_scene()
    intrinsics, points3D, points2D, pose = scene
    valid_mask = jp.ones(len(points2D), dtype=bool)
    key = jax.random.PRNGKey(0)
    RANSAC_args = (key, points3D, points2D, intrinsics, valid_mask, 200, 2.5)
    estimate = pnp.estimate_pose_RANSAC(*RANSAC_args)
    cv2_args = (np.array(points3D, np.float64),
                np.array(points2D, np.float64),
                np.array(intrinsics, np.float64), None)
    cv2_kwargs = {"reprojectionError": 2.5, "iterationsCount": 200,
                  "flags": cv2.SOLVEPNP_ITERATIVE}
    _, rvec, tvec, _ = cv2.solvePnPRansac(*cv2_args, **cv2_kwargs)
    rotation, _ = cv2.Rodrigues(rvec)
    pose_cv2 = paz.pinhole.to_affine_matrix(
        jp.array(rotation), jp.array(tvec[:, 0]))
    rotation_error = compute_rotation_degrees(estimate.pose, pose)
    rotation_error_cv2 = compute_rotation_degrees(pose_cv2, pose)
    assert rotation_error <= 1.1 * rotation_error_cv2 + 0.1
    translation_error = compute_translation_error(estimate.pose, pose)
    translation_error_cv2 = compute_translation_error(pose_cv2, pose)
    assert translation_error <= 1.1 * translation_error_cv2 + 0.005


def test_RANSAC_is_deterministic():
    scene = build_noisy_scene()
    intrinsics, points3D, points2D, pose = scene
    valid_mask = jp.ones(len(points2D), dtype=bool)
    key = jax.random.PRNGKey(0)
    RANSAC_args = (key, points3D, points2D, intrinsics, valid_mask, 200, 2.5)
    estimate_0 = pnp.estimate_pose_RANSAC(*RANSAC_args)
    estimate_1 = pnp.estimate_pose_RANSAC(*RANSAC_args)
    assert jp.array_equal(estimate_0.pose, estimate_1.pose)
    assert jp.array_equal(estimate_0.inliers, estimate_1.inliers)


def test_refine_pose_noise_free():
    intrinsics = build_intrinsics()
    points3D, pose = build_scene()
    points2D = project(intrinsics, pose, points3D)
    valid_mask = jp.ones(len(points2D), dtype=bool)
    translation_norm = jp.linalg.norm(pose[:3, 3])
    initial_pose = perturb_pose(pose, 2.0, 0.02 * translation_norm)
    refine_args = (initial_pose, points3D, points2D, intrinsics, valid_mask)
    refined = pnp.refine_pose(*refine_args, 15)
    assert bool(refined.valid)
    assert compute_rotation_degrees(refined.pose, pose) < 0.01
    assert compute_translation_error(refined.pose, pose) < 1e-4


def test_refine_pose_matches_scipy():
    intrinsics = build_intrinsics()
    points3D, pose = build_scene()
    points2D = project(intrinsics, pose, points3D)
    rng = np.random.default_rng(3)
    points2D = points2D + jp.array(rng.normal(0.0, 0.5, points2D.shape))
    valid_mask = jp.ones(len(points2D), dtype=bool)
    translation_norm = jp.linalg.norm(pose[:3, 3])
    initial_pose = perturb_pose(pose, 2.0, 0.02 * translation_norm)
    refine_args = (initial_pose, points3D, points2D, intrinsics, valid_mask)
    refined = pnp.refine_pose(*refine_args, 15)
    assert bool(refined.valid)
    RMSE_args = (refined.pose, points3D, points2D, intrinsics, valid_mask)
    RMSE = compute_RMSE(*RMSE_args)
    scipy_args = (initial_pose, points3D, points2D, intrinsics)
    RMSE_scipy = refine_with_scipy(*scipy_args)
    assert float(RMSE) <= 1.05 * RMSE_scipy


def test_refine_pose_huber_beats_plain_on_outliers():
    intrinsics = build_intrinsics()
    points3D, pose = build_scene(seed=2, num_points=100)
    points2D = project(intrinsics, pose, points3D)
    rng = np.random.default_rng(7)
    points2D = points2D + jp.array(rng.normal(0.0, 0.5, points2D.shape))
    shifts = jp.array(rng.uniform(30.0, 80.0, (10, 2)))
    points2D = points2D.at[:10].add(shifts)
    valid_mask = jp.ones(len(points2D), dtype=bool)
    translation_norm = jp.linalg.norm(pose[:3, 3])
    initial_pose = perturb_pose(pose, 2.0, 0.02 * translation_norm)
    refine_args = (initial_pose, points3D, points2D, intrinsics, valid_mask)
    plain = pnp.refine_pose(*refine_args, 15)
    huber = pnp.refine_pose_huber(*refine_args, 15, 2.0)
    assert bool(huber.valid)
    plain_rotation = compute_rotation_degrees(plain.pose, pose)
    huber_rotation = compute_rotation_degrees(huber.pose, pose)
    assert huber_rotation < plain_rotation
    plain_translation = compute_translation_error(plain.pose, pose)
    huber_translation = compute_translation_error(huber.pose, pose)
    assert huber_translation < plain_translation


def test_huber_weights():
    from paz.optimization.robust import huber_weights
    residual_norms = jp.array([0.0, 1.0, 2.0, 8.0])
    weights = huber_weights(residual_norms, 2.0)
    assert jp.allclose(weights, jp.array([1.0, 1.0, 1.0, 0.25]))


def test_insufficient_points_are_invalid():
    intrinsics = build_intrinsics()
    points3D, pose = build_scene()
    points2D = project(intrinsics, pose, points3D)
    valid_mask = jp.arange(len(points2D)) < 5
    estimate = pnp.solve_DLT(points3D, points2D, intrinsics, valid_mask)
    assert not bool(estimate.valid)
    assert bool(jp.all(jp.isfinite(estimate.pose)))
    key = jax.random.PRNGKey(0)
    RANSAC_args = (key, points3D, points2D, intrinsics, valid_mask, 50, 2.5)
    RANSAC_estimate = pnp.estimate_pose_RANSAC(*RANSAC_args)
    assert not bool(RANSAC_estimate.valid)
    assert bool(jp.all(jp.isfinite(RANSAC_estimate.pose)))


def test_coplanar_points_are_invalid():
    intrinsics = build_intrinsics()
    points3D, pose = build_scene()
    points3D = points3D.at[:, 2].set(6.0)
    points2D = project(intrinsics, pose, points3D)
    valid_mask = jp.ones(len(points2D), dtype=bool)
    estimate = pnp.solve_DLT(points3D, points2D, intrinsics, valid_mask)
    assert not bool(estimate.valid)
    assert bool(jp.all(jp.isfinite(estimate.pose)))


def test_RANSAC_jit_does_not_recompile():
    scene = build_noisy_scene()
    intrinsics, points3D, points2D, pose = scene
    valid_mask = jp.ones(len(points2D), dtype=bool)
    jitted = jax.jit(pnp.estimate_pose_RANSAC, static_argnums=(5,))
    jitted(jax.random.PRNGKey(0), points3D, points2D, intrinsics,
           valid_mask, 50, 2.5)
    jitted(jax.random.PRNGKey(1), points3D, points2D, intrinsics,
           valid_mask, 50, 3.0)
    assert jitted._cache_size() == 1


def test_refine_pose_jit_does_not_recompile():
    intrinsics = build_intrinsics()
    points3D, pose = build_scene()
    points2D = project(intrinsics, pose, points3D)
    valid_mask = jp.ones(len(points2D), dtype=bool)
    jitted = jax.jit(pnp.refine_pose, static_argnums=(5,))
    jitted(pose, points3D, points2D, intrinsics, valid_mask, 10)
    initial_pose = perturb_pose(pose, 1.0, 0.01)
    jitted(initial_pose, points3D, points2D, intrinsics, valid_mask, 10)
    assert jitted._cache_size() == 1
