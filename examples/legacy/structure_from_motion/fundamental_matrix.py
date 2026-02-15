import jax.numpy as jp
import paz


def compute_scale(points):
    num_points, dimension = points.shape
    mean = jp.mean(points, axis=0)
    stdv = jp.std(points, axis=0)
    scale_x = jp.sqrt(2) / stdv[0]
    scale_y = jp.sqrt(2) / stdv[1]
    return jp.array(
        [
            [scale_x, 0, -scale_x * mean[0]],
            [0, scale_y, -scale_y * mean[1]],
            [0, 0, 1.0],
        ]
    )


def build_A(normalized_points_1, normalized_points_2):
    x1, y1 = normalized_points_1[:, 0], normalized_points_1[:, 1]
    x2, y2 = normalized_points_2[:, 0], normalized_points_2[:, 1]
    ones = jp.ones(len(normalized_points_1))
    A = [x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones]
    A = jp.stack(A, axis=1)
    return A


def solve_fundamental_matrix(A):
    U, S, V = jp.linalg.svd(A)
    F_normalized = V[-1, :].reshape(3, 3)
    return F_normalized


def enforce_rank_2_constraint(F_normalized):
    U, S, V = jp.linalg.svd(F_normalized)
    S = S.at[-1].set(0)
    F_normalized = U @ jp.diag(S) @ V
    return F_normalized


def denormalize_fundamental_matrix(F_normalized, transform_2, transform_1):
    return transform_2.T @ F_normalized @ transform_1


def compute_fundamental_matrix(points2D_1, points2D_2):
    """Compute the fundamental matrix using the 8 point algorithm.

    # Arguments
        points2D_1: Array of shape (num_points, 2) in first image.
        points2D_2: Array of shape (num_points, 2) in second image.

    # Returns
        F: Array of shape (3, 3) representing the fundamental matrix.
    """
    transform_1 = compute_scale(points2D_1)
    transform_2 = compute_scale(points2D_2)
    points2D_1 = paz.points2D.transform(points2D_1, transform_1)
    points2D_2 = paz.points2D.transform(points2D_2, transform_2)
    A = build_A(points2D_1, points2D_2)
    F_normalized = solve_fundamental_matrix(A)
    F_normalized = enforce_rank_2_constraint(F_normalized)
    F = denormalize_fundamental_matrix(F_normalized, transform_1, transform_2)
    return F
