import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import cv2
import jax
import jax.numpy as jp
import numpy as np

import paz
from paz.backend import triangulation


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
    pose_B = paz.pinhole.to_affine_matrix(rotation, translation)
    return points3D, pose_B


def project(intrinsics, pose, points3D):
    camera_matrix = paz.pinhole.make_camera_matrix(intrinsics, pose)
    pixels = (camera_matrix @ paz.algebra.add_ones(points3D).T).T
    return pixels[:, :2] / pixels[:, 2:3]


def build_projected_scene():
    intrinsics = build_intrinsics()
    points3D, pose_B = build_scene()
    pose_A = jp.eye(4)
    P_A = paz.pinhole.make_camera_matrix(intrinsics, pose_A)
    P_B = paz.pinhole.make_camera_matrix(intrinsics, pose_B)
    points_A = project(intrinsics, pose_A, points3D)
    points_B = project(intrinsics, pose_B, points3D)
    return points3D, pose_A, pose_B, P_A, P_B, points_A, points_B


def test_triangulate_points_noise_free():
    scene = build_projected_scene()
    points3D, pose_A, pose_B, P_A, P_B, points_A, points_B = scene
    valid_mask = jp.ones(len(points_A), dtype=bool)
    recovered, valid = triangulation.triangulate_points(
        P_A, P_B, points_A, points_B, valid_mask)
    assert bool(jp.all(valid))
    assert jp.allclose(recovered, points3D, atol=1e-3)
    intrinsics = build_intrinsics()
    reprojected = project(intrinsics, pose_A, recovered)
    errors = jp.sum((reprojected - points_A) ** 2, axis=1)
    assert jp.sqrt(jp.mean(errors)) < 1e-3


def test_triangulate_points_matches_cv2():
    scene = build_projected_scene()
    points3D, pose_A, pose_B, P_A, P_B, points_A, points_B = scene
    valid_mask = jp.ones(len(points_A), dtype=bool)
    recovered, _ = triangulation.triangulate_points(
        P_A, P_B, points_A, points_B, valid_mask)
    cv2_args = (np.array(P_A, np.float64), np.array(P_B, np.float64),
                np.array(points_A, np.float64).T,
                np.array(points_B, np.float64).T)
    homogeneous = cv2.triangulatePoints(*cv2_args)
    recovered_cv2 = (homogeneous[:3] / homogeneous[3]).T
    assert np.allclose(recovered_cv2, np.array(recovered), atol=1e-4)


def test_triangulate_point_matches_batch():
    scene = build_projected_scene()
    points3D, pose_A, pose_B, P_A, P_B, points_A, points_B = scene
    point3D = triangulation.triangulate_point(
        P_A, P_B, points_A[0], points_B[0])
    valid_mask = jp.ones(len(points_A), dtype=bool)
    recovered, _ = triangulation.triangulate_points(
        P_A, P_B, points_A, points_B, valid_mask)
    assert jp.allclose(point3D, recovered[0], atol=1e-5)


def test_triangulate_points_propagates_valid_mask():
    scene = build_projected_scene()
    points3D, pose_A, pose_B, P_A, P_B, points_A, points_B = scene
    valid_mask = jp.arange(len(points_A)) >= 10
    _, valid = triangulation.triangulate_points(
        P_A, P_B, points_A, points_B, valid_mask)
    assert not bool(jp.any(valid[:10]))
    assert bool(jp.all(valid[10:]))


def test_triangulate_points_flags_points_at_infinity():
    intrinsics = build_intrinsics()
    pose_A = jp.eye(4)
    translation = jp.array([1.0, 0.0, 0.0])
    pose_B = paz.pinhole.to_affine_matrix(jp.eye(3), translation)
    P_A = paz.pinhole.make_camera_matrix(intrinsics, pose_A)
    P_B = paz.pinhole.make_camera_matrix(intrinsics, pose_B)
    points_A = jp.array([[320.0, 240.0], [400.0, 200.0]])
    valid_mask = jp.ones(2, dtype=bool)
    _, valid = triangulation.triangulate_points(
        P_A, P_B, points_A, points_A, valid_mask)
    assert not bool(jp.any(valid))


def test_triangulate_points_jit_does_not_recompile():
    scene = build_projected_scene()
    points3D, pose_A, pose_B, P_A, P_B, points_A, points_B = scene
    valid_mask = jp.ones(len(points_A), dtype=bool)
    jitted = jax.jit(triangulation.triangulate_points)
    jitted(P_A, P_B, points_A, points_B, valid_mask)
    jitted(P_B, P_A, points_B, points_A, valid_mask)
    assert jitted._cache_size() == 1


def test_compute_cheirality():
    pose_A = jp.eye(4)
    translation = jp.array([-1.0, 0.0, 0.0])
    pose_B = paz.pinhole.to_affine_matrix(jp.eye(3), translation)
    points3D = jp.array([[0.5, 0.0, 1.0], [0.0, 0.0, -1.0]])
    in_front = triangulation.compute_cheirality(pose_A, pose_B, points3D)
    assert jp.array_equal(in_front, jp.array([True, False]))


def test_compute_cheirality_counts_both_cameras():
    pose_A = jp.eye(4)
    rotation = paz.SO3.rotation_y(jp.pi)
    translation = jp.array([0.0, 0.0, 4.0])
    pose_B = paz.pinhole.to_affine_matrix(rotation, translation)
    points3D = jp.array([[0.0, 0.0, 2.0], [0.0, 0.0, 6.0]])
    in_front = triangulation.compute_cheirality(pose_A, pose_B, points3D)
    assert jp.array_equal(in_front, jp.array([True, False]))


def test_compute_parallax():
    pose_A = jp.eye(4)
    translation = jp.array([-1.0, 0.0, 0.0])
    pose_B = paz.pinhole.to_affine_matrix(jp.eye(3), translation)
    points3D = jp.array([[0.5, 0.0, 1.0]])
    angles = triangulation.compute_parallax(pose_A, pose_B, points3D)
    assert jp.allclose(angles, jp.arccos(0.6), atol=1e-6)


def test_compute_parallax_zero_baseline():
    pose_A = jp.eye(4)
    points3D = jp.array([[0.5, 0.2, 3.0], [-1.0, 0.4, 2.0]])
    angles = triangulation.compute_parallax(pose_A, pose_A, points3D)
    assert jp.allclose(angles, 0.0, atol=1e-3)
