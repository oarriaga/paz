from collections import namedtuple

import jax
import jax.numpy as jp

from paz.backend import algebra
from paz.backend import pinhole
from paz.backend.epipolar import normalize_weighted_points
from paz.backend.lie import SE3
from paz.optimization.robust import huber_weights

PnPEstimate = namedtuple("PnPEstimate", ["pose", "valid"])

PoseEstimate = namedtuple(
    "PoseEstimate", ["pose", "inliers", "num_inliers", "valid"])

RefinedPose = namedtuple(
    "RefinedPose", ["pose", "cost", "valid", "num_iterations"])


def estimate_pose_RANSAC(key, points3D, points2D, intrinsics, valid_mask,
                         num_hypotheses, threshold):
    valid_mask = valid_mask.astype(bool)
    weights = valid_mask.astype(points2D.dtype)
    probabilities = weights / jp.sum(weights)
    num_points = len(points2D)

    def find_inliers(pose):
        residual_args = (pose, points3D, points2D, intrinsics, valid_mask)
        residuals = compute_reprojection_residuals(*residual_args)
        errors = jp.linalg.norm(residuals, axis=1)
        inliers = (errors < threshold) & valid_mask
        return inliers, jp.sum(inliers)

    def step(state, key):
        best_pose, best_inliers, best_count = state
        sample_args = (key, num_points, (6,), False, probabilities)
        indices = jax.random.choice(*sample_args)
        sample_mask = jp.ones(6, dtype=bool)
        estimate = solve_DLT(
            points3D[indices], points2D[indices], intrinsics, sample_mask)
        inliers, count = find_inliers(estimate.pose)

        def update():
            return estimate.pose, inliers, count

        def keep():
            return state

        better = estimate.valid & (count > best_count)
        return jax.lax.cond(better, update, keep), None

    def refit(state):
        # Second round refines on the grown consensus set, since one
        # round from a small minimal-sample mask is not converged.
        pose, inliers, count = state
        refine_args = (pose, points3D, points2D, intrinsics, inliers, 10)
        refined = refine_pose(*refine_args)
        refit_inliers, refit_count = find_inliers(refined.pose)
        keep_refit = refined.valid & (refit_count >= count)
        pose = jp.where(keep_refit, refined.pose, pose)
        inliers = jp.where(keep_refit, refit_inliers, inliers)
        count = jp.where(keep_refit, refit_count, count)
        return pose, inliers, count

    inliers = jp.zeros(num_points, dtype=bool)
    state = (jp.eye(4, dtype=points2D.dtype), inliers, jp.sum(inliers))
    keys = jax.random.split(key, num_hypotheses)
    state, _ = jax.lax.scan(step, state, keys)
    pose, inliers, count = refit(refit(state))
    return PoseEstimate(pose, inliers, count, count >= 6)


def solve_DLT(points3D, points2D, intrinsics, valid_mask):
    weights = valid_mask.astype(points2D.dtype)
    resection = compute_weighted_camera_matrix(points3D, points2D, weights)
    camera_matrix, conditioning = resection
    camera_matrix = orient_camera_matrix(camera_matrix, points3D, weights)
    pose = factor_camera_matrix(camera_matrix, intrinsics)
    enough = jp.sum(valid_mask.astype(bool)) >= 6
    finite = jp.all(jp.isfinite(pose))
    well_conditioned = conditioning > 1e-4
    return PnPEstimate(pose, enough & finite & well_conditioned)


def compute_weighted_camera_matrix(points3D, points2D, weights):
    transform2D, normalized2D = normalize_weighted_points(points2D, weights)
    transform3D, normalized3D = normalize_weighted_points3D(points3D, weights)
    design_matrix = build_design_matrix(normalized3D, normalized2D)
    row_weights = jp.concatenate([weights, weights])
    design_matrix = jp.sqrt(row_weights)[:, None] * design_matrix
    _, S, Vt = jp.linalg.svd(design_matrix)
    conditioning = S[10] / S[0]
    camera_matrix = Vt[-1].reshape(3, 4)
    camera_matrix = jp.linalg.inv(transform2D) @ camera_matrix @ transform3D
    return camera_matrix, conditioning


def normalize_weighted_points3D(points3D, weights):
    total_weight = jp.sum(weights)
    centroid = jp.sum(weights[:, None] * points3D, axis=0) / total_weight
    squared_distances = jp.sum((points3D - centroid) ** 2, axis=1)
    rms_distance = jp.sqrt(jp.sum(weights * squared_distances) / total_weight)
    scale = jp.sqrt(3.0) / rms_distance
    zero, one = jp.zeros_like(scale), jp.ones_like(scale)
    transform = jp.array(
        [[scale, zero, zero, -scale * centroid[0]],
         [zero, scale, zero, -scale * centroid[1]],
         [zero, zero, scale, -scale * centroid[2]],
         [zero, zero, zero, one]])
    return transform, algebra.transform_points(transform, points3D)


def build_design_matrix(points3D, points2D):
    homogeneous = algebra.add_ones(points3D)
    zeros = jp.zeros_like(homogeneous)
    x = points2D[:, 0:1]
    y = points2D[:, 1:2]
    rows_x = jp.concatenate([homogeneous, zeros, -x * homogeneous], axis=1)
    rows_y = jp.concatenate([zeros, homogeneous, -y * homogeneous], axis=1)
    return jp.concatenate([rows_x, rows_y], axis=0)


def orient_camera_matrix(camera_matrix, points3D, weights):
    # Fix the projective sign so valid points get positive depth.
    depths = algebra.add_ones(points3D) @ camera_matrix[2]
    mean_depth = jp.sum(weights * depths) / jp.sum(weights)
    return jp.where(mean_depth < 0.0, -camera_matrix, camera_matrix)


def factor_camera_matrix(camera_matrix, intrinsics):
    motion = jp.linalg.inv(intrinsics) @ camera_matrix
    U, S, Vt = jp.linalg.svd(motion[:, :3])
    determinant = jp.linalg.det(U @ Vt)
    rotation = U @ jp.diag(jp.array([1.0, 1.0, determinant])) @ Vt
    translation = motion[:, 3] / jp.mean(S)
    return pinhole.to_affine_matrix(rotation, translation)


def compute_reprojection_residuals(pose, points3D, points2D, intrinsics,
                                   valid_mask):
    camera_matrix = pinhole.make_camera_matrix(intrinsics, pose)
    pixels = algebra.add_ones(points3D) @ camera_matrix.T
    depths = pixels[:, 2]
    safe_depths = jp.where(jp.abs(depths) < 1e-8, 1e-8, depths)
    residuals = pixels[:, :2] / safe_depths[:, None] - points2D
    return residuals * valid_mask.astype(residuals.dtype)[:, None]


def refine_pose(initial_pose, points3D, points2D, intrinsics, valid_mask,
                iterations):
    args = (initial_pose, points3D, points2D, intrinsics, valid_mask)
    return run_gauss_newton(*args, iterations, jp.ones_like)


def refine_pose_huber(initial_pose, points3D, points2D, intrinsics,
                      valid_mask, iterations, scale):
    def compute_weights(residual_norms):
        return huber_weights(residual_norms, scale)

    args = (initial_pose, points3D, points2D, intrinsics, valid_mask)
    return run_gauss_newton(*args, iterations, compute_weights)


def run_gauss_newton(initial_pose, points3D, points2D, intrinsics,
                     valid_mask, iterations, compute_weights):
    mask = valid_mask.astype(points2D.dtype)

    def compute_residuals(pose):
        residual_args = (pose, points3D, points2D, intrinsics, valid_mask)
        return compute_reprojection_residuals(*residual_args)

    def compute_cost(pose):
        errors = jp.linalg.norm(compute_residuals(pose), axis=1)
        weights = mask * compute_weights(errors)
        return jp.sum(weights * errors ** 2) / jp.sum(mask)

    def compute_step(pose):
        def residuals_of(delta):
            return compute_residuals(SE3.retract(pose, delta)).reshape(-1)

        zero_delta = jp.zeros(6, dtype=initial_pose.dtype)
        residuals = residuals_of(zero_delta)
        jacobian = jax.jacfwd(residuals_of)(zero_delta)
        errors = jp.linalg.norm(residuals.reshape(-1, 2), axis=1)
        point_weights = mask * compute_weights(errors)
        row_weights = jp.repeat(point_weights, 2)
        weighted_jacobian = row_weights[:, None] * jacobian
        JtJ = jacobian.T @ weighted_jacobian
        gradient = weighted_jacobian.T @ residuals
        damping = 1e-6 * jp.trace(JtJ) / 6.0
        delta = jp.linalg.solve(JtJ + damping * jp.eye(6), -gradient)
        return SE3.retract(pose, delta)

    def step(iteration, state):
        pose, cost = state
        candidate = compute_step(pose)
        candidate_cost = compute_cost(candidate)

        def take():
            return candidate, candidate_cost

        def keep():
            return state

        return jax.lax.cond(candidate_cost <= cost, take, keep)

    state = (initial_pose, compute_cost(initial_pose))
    pose, cost = jax.lax.fori_loop(0, iterations, step, state)
    valid = jp.all(jp.isfinite(pose)) & jp.isfinite(cost)
    return RefinedPose(pose, cost, valid, iterations)
