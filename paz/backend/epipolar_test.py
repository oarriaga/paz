import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import cv2
import jax
import jax.numpy as jp
import numpy as np
from jax.experimental import enable_x64

import paz
from paz.backend import epipolar


def build_intrinsics():
    return jp.array([[500.0, 0.0, 320.0],
                     [0.0, 500.0, 240.0],
                     [0.0, 0.0, 1.0]])


def build_scene(seed=0, num_points=60):
    rng = np.random.default_rng(seed)
    points3D = jp.array(rng.uniform(-1.0, 1.0, (num_points, 3)))
    points3D = points3D + jp.array([0.0, 0.0, 5.0])
    rotation = paz.SO3.rotation_y(0.15) @ paz.SO3.rotation_x(0.05)
    translation = jp.array([0.5, 0.1, 0.2])
    return points3D, rotation, translation


def project(intrinsics, pose, points3D):
    camera_matrix = paz.pinhole.make_camera_matrix(intrinsics, pose)
    pixels = (camera_matrix @ paz.algebra.add_ones(points3D).T).T
    return pixels[:, :2] / pixels[:, 2:3]


def project_pair(intrinsics_A, intrinsics_B, points3D, rotation, translation):
    pose_B = paz.pinhole.to_affine_matrix(rotation, translation)
    points_A = project(intrinsics_A, jp.eye(4), points3D)
    points_B = project(intrinsics_B, pose_B, points3D)
    return points_A, points_B


def compute_geodesic_degrees(rotation_A, rotation_B):
    cosine = (jp.trace(rotation_A @ rotation_B.T) - 1.0) / 2.0
    return jp.rad2deg(jp.arccos(jp.clip(cosine, -1.0, 1.0)))


def compute_angle_degrees(unit_vector_A, unit_vector_B):
    cosine = jp.dot(unit_vector_A, unit_vector_B)
    return jp.rad2deg(jp.arccos(jp.clip(cosine, -1.0, 1.0)))


def normalize_fundamental(fundamental_matrix):
    fundamental_matrix = np.array(fundamental_matrix, dtype=np.float64)
    fundamental_matrix = fundamental_matrix / np.linalg.norm(
        fundamental_matrix)
    largest = np.argmax(np.abs(fundamental_matrix))
    sign = np.sign(fundamental_matrix.flat[largest])
    return sign * fundamental_matrix


def test_normalize_points():
    points = jp.array(np.random.default_rng(0).uniform(0, 640, (30, 2)))
    transform, normalized = epipolar.normalize_points(points)
    recovered = paz.algebra.transform_points(transform, points)
    assert jp.allclose(recovered, normalized, atol=1e-5)
    assert jp.allclose(jp.mean(normalized, axis=0), 0.0, atol=1e-4)
    rms = jp.sqrt(jp.mean(jp.sum(normalized ** 2, axis=1)))
    assert jp.allclose(rms, jp.sqrt(2.0), atol=1e-4)


def test_fundamental_matrix_noise_free():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    F = epipolar.compute_fundamental_matrix(points_A, points_B)
    distances = epipolar.compute_sampson_distance(F, points_A, points_B)
    assert jp.median(distances) < 1e-4
    unit_F = jp.array(normalize_fundamental(F), dtype=points_A.dtype)
    homogeneous_A = paz.algebra.add_ones(points_A)
    homogeneous_B = paz.algebra.add_ones(points_B)
    residuals = jp.sum(homogeneous_B * (unit_F @ homogeneous_A.T).T, axis=1)
    assert jp.max(jp.abs(residuals)) < 1e-5


def test_fundamental_matrix_matches_cv2():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    F = epipolar.compute_fundamental_matrix(points_A, points_B)
    cv2_args = (np.array(points_A, np.float64),
                np.array(points_B, np.float64), cv2.FM_8POINT)
    F_cv2, _ = cv2.findFundamentalMat(*cv2_args)
    assert np.allclose(
        normalize_fundamental(F), normalize_fundamental(F_cv2), atol=1e-5)


def test_fundamental_matrix_float64():
    with enable_x64():
        intrinsics = build_intrinsics().astype(jp.float64)
        points3D, rotation, translation = build_scene()
        points3D = points3D.astype(jp.float64)
        points_A, points_B = project_pair(
            intrinsics, intrinsics, points3D, rotation, translation)
        F = epipolar.compute_fundamental_matrix(points_A, points_B)
        assert F.dtype == jp.float64
        distances = epipolar.compute_sampson_distance(F, points_A, points_B)
        assert jp.median(distances) < 1e-10


def test_RANSAC_rejects_outliers():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    points_B = points_B.at[:12].add(60.0)
    valid_mask = jp.ones(len(points_A), dtype=bool)
    key = jax.random.PRNGKey(0)
    RANSAC_args = (key, points_A, points_B, valid_mask, 200, 1.0)
    estimate = epipolar.estimate_fundamental_matrix_RANSAC(*RANSAC_args)
    assert bool(estimate.valid)
    inliers = np.array(estimate.inliers)
    assert inliers[:12].sum() == 0
    assert inliers[12:].sum() >= int(0.95 * 48)
    assert int(estimate.num_inliers) == inliers.sum()


def test_RANSAC_is_deterministic():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    points_B = points_B.at[:12].add(60.0)
    valid_mask = jp.ones(len(points_A), dtype=bool)
    key = jax.random.PRNGKey(0)
    RANSAC_args = (key, points_A, points_B, valid_mask, 200, 1.0)
    estimate_0 = epipolar.estimate_fundamental_matrix_RANSAC(*RANSAC_args)
    estimate_1 = epipolar.estimate_fundamental_matrix_RANSAC(*RANSAC_args)
    assert jp.array_equal(
        estimate_0.fundamental_matrix, estimate_1.fundamental_matrix)
    assert jp.array_equal(estimate_0.inliers, estimate_1.inliers)


def test_RANSAC_insufficient_correspondences():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    valid_mask = jp.arange(len(points_A)) < 5
    key = jax.random.PRNGKey(0)
    RANSAC_args = (key, points_A, points_B, valid_mask, 50, 1.0)
    estimate = epipolar.estimate_fundamental_matrix_RANSAC(*RANSAC_args)
    assert not bool(estimate.valid)
    assert bool(jp.all(jp.isfinite(estimate.fundamental_matrix)))


def test_RANSAC_all_outliers_stays_finite():
    key = jax.random.PRNGKey(3)
    key_A, key_B = jax.random.split(key)
    points_A = jax.random.uniform(key_A, (40, 2)) * 640.0
    points_B = jax.random.uniform(key_B, (40, 2)) * 640.0
    valid_mask = jp.ones(len(points_A), dtype=bool)
    RANSAC_args = (key, points_A, points_B, valid_mask, 50, 1e-6)
    estimate = epipolar.estimate_fundamental_matrix_RANSAC(*RANSAC_args)
    assert not bool(estimate.valid)
    assert bool(jp.all(jp.isfinite(estimate.fundamental_matrix)))


def test_RANSAC_jit_does_not_recompile():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    valid_mask = jp.ones(len(points_A), dtype=bool)
    jitted = jax.jit(
        epipolar.estimate_fundamental_matrix_RANSAC, static_argnums=(4,))
    jitted(jax.random.PRNGKey(0), points_A, points_B, valid_mask, 50, 1.0)
    jitted(jax.random.PRNGKey(1), points_A, points_B, valid_mask, 50, 2.0)
    assert jitted._cache_size() == 1


def test_recover_relative_pose():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    F = epipolar.compute_fundamental_matrix(points_A, points_B)
    E = epipolar.compute_essential_matrix(F, intrinsics, intrinsics)
    valid_mask = jp.ones(len(points_A), dtype=bool)
    pose_args = (E, intrinsics, intrinsics, points_A, points_B, valid_mask)
    pose = epipolar.recover_relative_pose(*pose_args)
    assert bool(pose.valid)
    assert int(pose.num_in_front) == len(points_A)
    assert compute_geodesic_degrees(pose.rotation, rotation) < 0.05
    direction = translation / jp.linalg.norm(translation)
    assert compute_angle_degrees(pose.translation, direction) < 0.1


def test_recover_relative_pose_different_intrinsics():
    intrinsics_A = build_intrinsics()
    intrinsics_B = jp.array([[450.0, 0.0, 300.0],
                             [0.0, 460.0, 250.0],
                             [0.0, 0.0, 1.0]])
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics_A, intrinsics_B, points3D, rotation, translation)
    F = epipolar.compute_fundamental_matrix(points_A, points_B)
    E = epipolar.compute_essential_matrix(F, intrinsics_A, intrinsics_B)
    valid_mask = jp.ones(len(points_A), dtype=bool)
    pose_args = (E, intrinsics_A, intrinsics_B, points_A, points_B,
                 valid_mask)
    pose = epipolar.recover_relative_pose(*pose_args)
    assert bool(pose.valid)
    assert compute_geodesic_degrees(pose.rotation, rotation) < 0.05
    direction = translation / jp.linalg.norm(translation)
    assert compute_angle_degrees(pose.translation, direction) < 0.1


def test_recover_relative_pose_matches_cv2():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    F = epipolar.compute_fundamental_matrix(points_A, points_B)
    E = epipolar.compute_essential_matrix(F, intrinsics, intrinsics)
    valid_mask = jp.ones(len(points_A), dtype=bool)
    pose_args = (E, intrinsics, intrinsics, points_A, points_B, valid_mask)
    pose = epipolar.recover_relative_pose(*pose_args)
    cv2_args = (np.array(E, np.float64), np.array(points_A, np.float64),
                np.array(points_B, np.float64),
                np.array(intrinsics, np.float64))
    _, rotation_cv2, translation_cv2, _ = cv2.recoverPose(*cv2_args)
    assert np.allclose(rotation_cv2, np.array(pose.rotation), atol=1e-4)
    assert np.allclose(
        translation_cv2[:, 0], np.array(pose.translation), atol=1e-4)


def test_pure_rotation_is_invalid():
    intrinsics = build_intrinsics()
    points3D, rotation, _ = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, jp.zeros(3))
    F = epipolar.compute_fundamental_matrix(points_A, points_B)
    E = epipolar.compute_essential_matrix(F, intrinsics, intrinsics)
    valid_mask = jp.ones(len(points_A), dtype=bool)
    pose_args = (E, intrinsics, intrinsics, points_A, points_B, valid_mask)
    pose = epipolar.recover_relative_pose(*pose_args)
    assert not bool(pose.valid)


def test_planar_scene_is_invalid():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points3D = points3D.at[:, 2].set(5.0 + 0.01 * points3D[:, 2])
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    F = epipolar.compute_fundamental_matrix(points_A, points_B)
    E = epipolar.compute_essential_matrix(F, intrinsics, intrinsics)
    valid_mask = jp.ones(len(points_A), dtype=bool)
    pose_args = (E, intrinsics, intrinsics, points_A, points_B, valid_mask)
    pose = epipolar.recover_relative_pose(*pose_args)
    assert not bool(pose.valid)


def test_essential_constraints():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    F = epipolar.compute_fundamental_matrix(points_A, points_B)
    E = epipolar.compute_essential_matrix(F, intrinsics, intrinsics)
    singular_values = jp.linalg.svd(E, compute_uv=False)
    assert jp.allclose(singular_values[0], singular_values[1], rtol=1e-5)
    assert jp.abs(singular_values[2]) < 1e-5 * singular_values[0]


def test_decompose_essential_matrix_gives_proper_rotations():
    intrinsics = build_intrinsics()
    points3D, rotation, translation = build_scene()
    points_A, points_B = project_pair(
        intrinsics, intrinsics, points3D, rotation, translation)
    F = epipolar.compute_fundamental_matrix(points_A, points_B)
    E = epipolar.compute_essential_matrix(F, intrinsics, intrinsics)
    rotations, translations = epipolar.decompose_essential_matrix(E)
    assert rotations.shape == (4, 3, 3)
    assert translations.shape == (4, 3)
    determinants = jax.vmap(jp.linalg.det)(rotations)
    assert jp.allclose(determinants, 1.0, atol=1e-4)
