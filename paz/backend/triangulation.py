import jax
import jax.numpy as jp

from paz.backend import algebra
from paz.backend.lie import SE3


def triangulate_points(P_A, P_B, points_A, points_B, valid_mask):
    solve = jax.vmap(solve_homogeneous_point, in_axes=(None, None, 0, 0))
    homogeneous = solve(P_A, P_B, points_A, points_B)
    w = homogeneous[:, 3]
    near_zero_w = jp.abs(w) < 1e-6
    safe_w = jp.where(near_zero_w, jp.ones_like(w), w)
    points3D = homogeneous[:, :3] / safe_w[:, None]
    finite = jp.all(jp.isfinite(points3D), axis=1)
    valid = valid_mask.astype(bool) & finite & ~near_zero_w
    return points3D, valid


def triangulate_point(P_A, P_B, point_A, point_B):
    homogeneous = solve_homogeneous_point(P_A, P_B, point_A, point_B)
    return homogeneous[:3] / homogeneous[3]


def solve_homogeneous_point(P_A, P_B, point_A, point_B):
    design_matrix = build_design_matrix(P_A, P_B, point_A, point_B)
    _, _, Vt = jp.linalg.svd(design_matrix)
    return Vt[-1]


def build_design_matrix(P_A, P_B, point_A, point_B):
    return jp.stack(
        [point_A[0] * P_A[2] - P_A[0], point_A[1] * P_A[2] - P_A[1],
         point_B[0] * P_B[2] - P_B[0], point_B[1] * P_B[2] - P_B[1]])


def compute_cheirality(pose_A, pose_B, points3D):
    depths_A = algebra.transform_points(pose_A, points3D)[:, 2]
    depths_B = algebra.transform_points(pose_B, points3D)[:, 2]
    return (depths_A > 0.0) & (depths_B > 0.0)


def compute_parallax(pose_A, pose_B, points3D):
    rays_A = compute_camera_center(pose_A) - points3D
    rays_B = compute_camera_center(pose_B) - points3D
    dots = jp.sum(rays_A * rays_B, axis=1)
    norms = jp.linalg.norm(rays_A, axis=1) * jp.linalg.norm(rays_B, axis=1)
    cosines = jp.clip(dots / norms, -1.0, 1.0)
    return jp.arccos(cosines)


def compute_camera_center(pose):
    return SE3.get_position_vector(SE3.invert(pose))
