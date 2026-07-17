from collections import namedtuple

import jax
import jax.numpy as jp

from paz.backend import algebra
from paz.backend import pinhole
from paz.backend import triangulation

FundamentalEstimate = namedtuple(
    "FundamentalEstimate",
    ["fundamental_matrix", "inliers", "num_inliers", "valid"])

RelativePose = namedtuple(
    "RelativePose", ["rotation", "translation", "num_in_front", "valid"])


def estimate_fundamental_matrix_RANSAC(key, points_A, points_B, valid_mask,
                                       num_hypotheses, threshold):
    valid_mask = valid_mask.astype(bool)
    weights = valid_mask.astype(points_A.dtype)
    probabilities = weights / jp.sum(weights)
    num_points = len(points_A)

    def find_inliers(fundamental_matrix):
        distances = compute_sampson_distance(
            fundamental_matrix, points_A, points_B)
        inliers = (distances < threshold) & valid_mask
        return inliers, jp.sum(inliers)

    def step(state, key):
        best_F, best_inliers, best_count = state
        sample_args = (key, num_points, (8,), False, probabilities)
        indices = jax.random.choice(*sample_args)
        F = compute_fundamental_matrix(points_A[indices], points_B[indices])
        inliers, count = find_inliers(F)

        def update():
            return F, inliers, count

        def keep():
            return state

        return jax.lax.cond(count > best_count, update, keep), None

    inliers = jp.zeros(num_points, dtype=bool)
    state = (jp.eye(3, dtype=points_A.dtype), inliers, jp.sum(inliers))
    keys = jax.random.split(key, num_hypotheses)
    (F, inliers, count), _ = jax.lax.scan(step, state, keys)
    # one guarded refit is not converged when the consensus set grows;
    # a second round refits on the grown set (LO-RANSAC, as in pnp)
    for _ in range(2):
        F, inliers, count = refit_fundamental_matrix(
            points_A, points_B, F, inliers, count, find_inliers)
    return FundamentalEstimate(F, inliers, count, count >= 8)


def refit_fundamental_matrix(points_A, points_B, F, inliers, count,
                             find_inliers):
    refit_weights = inliers.astype(points_A.dtype)
    refit_F = compute_weighted_fundamental_matrix(
        points_A, points_B, refit_weights)
    refit_inliers, refit_count = find_inliers(refit_F)
    # a zero-inlier refit divides by zero total weight and returns NaN;
    # only accept a finite refit that keeps the consensus set
    finite = jp.all(jp.isfinite(refit_F))
    keep_refit = finite & (refit_count >= count)
    F = jp.where(keep_refit, refit_F, F)
    inliers = jp.where(keep_refit, refit_inliers, inliers)
    count = jp.where(keep_refit, refit_count, count)
    return F, inliers, count


def compute_fundamental_matrix(points_A, points_B):
    weights = jp.ones(len(points_A), dtype=points_A.dtype)
    return compute_weighted_fundamental_matrix(points_A, points_B, weights)


def compute_weighted_fundamental_matrix(points_A, points_B, weights):
    normalized = build_normalized_design(points_A, points_B, weights)
    transform_A, transform_B, design_matrix = normalized
    _, _, Vt = jp.linalg.svd(design_matrix)
    fundamental_matrix = enforce_rank_two(Vt[-1].reshape(3, 3))
    return transform_B.T @ fundamental_matrix @ transform_A


def compute_conditioning(points_A, points_B, weights):
    """Ratio of the design matrix's 8th to 1st singular value. Near zero
    for degenerate correspondences (planar scenes or pure rotation), where
    the fundamental matrix is not unique."""
    _, _, design_matrix = build_normalized_design(points_A, points_B, weights)
    S = jp.linalg.svd(design_matrix, compute_uv=False)
    return S[7] / S[0]


def build_normalized_design(points_A, points_B, weights):
    transform_A, points_A = normalize_weighted_points(points_A, weights)
    transform_B, points_B = normalize_weighted_points(points_B, weights)
    design_matrix = build_design_matrix(points_A, points_B)
    design_matrix = jp.sqrt(weights)[:, None] * design_matrix
    return transform_A, transform_B, design_matrix


def build_design_matrix(points_A, points_B):
    x_A, y_A = points_A[:, 0], points_A[:, 1]
    x_B, y_B = points_B[:, 0], points_B[:, 1]
    ones = jp.ones_like(x_A)
    columns = (x_B * x_A, x_B * y_A, x_B, y_B * x_A,
               y_B * y_A, y_B, x_A, y_A, ones)
    return jp.stack(columns, axis=1)


def enforce_rank_two(matrix):
    U, S, Vt = jp.linalg.svd(matrix)
    S = S.at[2].set(0.0)
    return U @ jp.diag(S) @ Vt


def normalize_points(points):
    weights = jp.ones(len(points), dtype=points.dtype)
    return normalize_weighted_points(points, weights)


def normalize_weighted_points(points, weights):
    total_weight = jp.sum(weights)
    centroid = jp.sum(weights[:, None] * points, axis=0) / total_weight
    squared_distances = jp.sum((points - centroid) ** 2, axis=1)
    rms_distance = jp.sqrt(jp.sum(weights * squared_distances) / total_weight)
    scale = jp.sqrt(2.0) / rms_distance
    zero, one = jp.zeros_like(scale), jp.ones_like(scale)
    transform = jp.array([[scale, zero, -scale * centroid[0]],
                          [zero, scale, -scale * centroid[1]],
                          [zero, zero, one]])
    return transform, algebra.transform_points(transform, points)


def compute_sampson_distance(fundamental_matrix, points_A, points_B):
    points_A = algebra.add_ones(points_A)
    points_B = algebra.add_ones(points_B)
    lines_B = fundamental_matrix @ points_A.T
    lines_A = fundamental_matrix.T @ points_B.T
    numerator = jp.sum(points_B * lines_B.T, axis=1)
    lines_B_norm = jp.sum(lines_B[:2] ** 2, axis=0)
    lines_A_norm = jp.sum(lines_A[:2] ** 2, axis=0)
    return jp.abs(numerator) / jp.sqrt(lines_B_norm + lines_A_norm)


def compute_essential_matrix(fundamental_matrix, intrinsics_A, intrinsics_B):
    essential_matrix = intrinsics_B.T @ fundamental_matrix @ intrinsics_A
    return enforce_essential_constraints(essential_matrix)


def enforce_essential_constraints(essential_matrix):
    U, S, Vt = jp.linalg.svd(essential_matrix)
    mean_singular_value = (S[0] + S[1]) / 2.0
    zero = jp.zeros_like(mean_singular_value)
    S = jp.stack([mean_singular_value, mean_singular_value, zero])
    return U @ jp.diag(S) @ Vt


def decompose_essential_matrix(essential_matrix):
    """Decomposes E into the 4 candidate (rotation, translation) poses.

    SVD does not guarantee proper rotations (det=+1), so U and V's last
    column/row are flipped when their determinant is negative, following
    Hartley & Zisserman section 9.6.2.
    """
    U, _, Vt = jp.linalg.svd(essential_matrix)
    U = jp.where(jp.linalg.det(U) < 0.0, U.at[:, -1].multiply(-1.0), U)
    Vt = jp.where(jp.linalg.det(Vt) < 0.0, Vt.at[-1, :].multiply(-1.0), Vt)
    W = jp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotation_0, rotation_1 = U @ W @ Vt, U @ W.T @ Vt
    translation_0, translation_1 = U[:, 2], -U[:, 2]
    rotations = jp.stack([rotation_0, rotation_0, rotation_1, rotation_1])
    translations = jp.stack(
        [translation_0, translation_1, translation_0, translation_1])
    return rotations, translations


def recover_relative_pose(essential_matrix, intrinsics_A, intrinsics_B,
                          points_A, points_B, valid_mask):
    """Selects the (R, t) candidate with the most points in front of both
    cameras (cheirality check), as in Hartley & Zisserman section 9.6.3."""
    rotations, translations = decompose_essential_matrix(essential_matrix)
    pose_A = jp.eye(4, dtype=essential_matrix.dtype)
    P_A = pinhole.make_camera_matrix(intrinsics_A, pose_A)

    def count_points_in_front(rotation, translation):
        pose_B = pinhole.to_affine_matrix(rotation, translation)
        P_B = pinhole.make_camera_matrix(intrinsics_B, pose_B)
        triangulate_args = (P_A, P_B, points_A, points_B, valid_mask)
        points3D, points_valid = triangulation.triangulate_points(
            *triangulate_args)
        in_front = triangulation.compute_cheirality(pose_A, pose_B, points3D)
        return jp.sum(in_front & points_valid)

    counts = jax.vmap(count_points_in_front)(rotations, translations)
    best = jp.argmax(counts)
    translation = translations[best]
    translation = translation / jp.linalg.norm(translation)
    sorted_counts = jp.sort(counts)
    best_count, second_count = sorted_counts[3], sorted_counts[2]
    dominates = best_count > 2 * second_count
    enough = best_count > 0.75 * jp.sum(valid_mask.astype(bool))
    weights = valid_mask.astype(points_A.dtype)
    conditioning = compute_conditioning(points_A, points_B, weights)
    well_conditioned = conditioning > 1e-3
    valid = dominates & enough & well_conditioned
    return RelativePose(rotations[best], translation, best_count, valid)
