import jax
import jax.numpy as jp
import paz


def center_and_normalize_points(points):
    centroid = jp.mean(points, axis=0)
    centered = points - centroid
    rms_distance = jp.sqrt(jp.sum(centered ** 2) / len(points))
    scale = jp.sqrt(2.0) / rms_distance
    transform = jp.array([[scale, 0.0, -scale * centroid[0]],
                         [0.0, scale, -scale * centroid[1]],
                         [0.0, 0.0, 1.0]])
    normalized_points = paz.algebra.transform_points(transform, points)
    return transform, normalized_points


def compute_fundamental_matrix(points1, points2):
    """Normalized eight-point algorithm. Works for any N >= 8 points."""
    T1, points1 = center_and_normalize_points(points1)
    T2, points2 = center_and_normalize_points(points2)
    x1, y1 = points1[:, 0], points1[:, 1]
    x2, y2 = points2[:, 0], points2[:, 1]
    ones = jp.ones_like(x1)
    A = jp.stack([x1*x2, x2*y1, x2, y2*x1, y1*y2, y2, x1, y1, ones], axis=1)
    _, _, Vt = jp.linalg.svd(A)
    fundamental_matrix = Vt[-1].reshape(3, 3)

    U, S, Vt = jp.linalg.svd(fundamental_matrix)
    S = S.at[-1].set(0.0)
    fundamental_matrix = U @ jp.diag(S) @ Vt
    return T2.T @ fundamental_matrix @ T1


def compute_sampson_distance(fundamental_matrix, points1, points2):
    points1 = paz.algebra.add_ones(points1)
    points2 = paz.algebra.add_ones(points2)
    line1 = fundamental_matrix @ points1.T
    line2 = fundamental_matrix.T @ points2.T
    numerator = jp.sum(points2 * line1.T, axis=1)
    line1_norm = jp.sum(line1[:2] ** 2, axis=0)
    line2_norm = jp.sum(line2[:2] ** 2, axis=0)
    return jp.abs(numerator) / jp.sqrt(line1_norm + line2_norm)


def sample_correspondences(key, points1, points2, num_points):
    indices = jax.random.choice(key, len(points1), (num_points,), False)
    return points1[indices], points2[indices]


def estimate_fundamental_matrix_RANSAC(key, points1, points2, num_points=8,
                                       steps=1000, threshold=0.5):
    """Fit fundamental matrix using RANSAC, following paz.plane.fit_RANSAC."""

    def find_inliers(fundamental_matrix):
        distances = compute_sampson_distance(
            fundamental_matrix, points1, points2)
        mask = distances < threshold
        return mask, jp.sum(mask)

    def step(state, key):
        best_F, best_count, best_mask = state
        sample1, sample2 = sample_correspondences(
            key, points1, points2, num_points)
        F = compute_fundamental_matrix(sample1, sample2)
        mask, count = find_inliers(F)
        should_update = count > best_count

        def update():
            return F, count, mask

        def keep():
            return state

        return jax.lax.cond(should_update, update, keep), None

    state = jp.eye(3), 0, jp.zeros(len(points1), dtype=bool)
    keys = jax.random.split(key, steps)
    (fundamental_matrix, _, inlier_mask), _ = jax.lax.scan(step, state, keys)
    return fundamental_matrix, inlier_mask


def compute_essential_matrix(fundamental_matrix, camera_intrinsics):
    weighted_fundamental_matrix = camera_intrinsics.T @ fundamental_matrix
    essential_matrix = weighted_fundamental_matrix @ camera_intrinsics
    U, S, Vt = jp.linalg.svd(essential_matrix)
    S = S.at[2].set(0.0)
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
    rotation1, rotation2 = U @ W @ Vt, U @ W.T @ Vt
    translation1, translation2 = U[:, 2], -U[:, 2]
    rotations = jp.stack([rotation1, rotation1, rotation2, rotation2])
    translations = jp.stack(
        [translation1, translation2, translation1, translation2])
    return rotations, translations


def build_design_matrix(P1, P2, point1, point2):
    return jp.stack([point1[0] * P1[2] - P1[0], point1[1] * P1[2] - P1[1],
                     point2[0] * P2[2] - P2[0], point2[1] * P2[2] - P2[1]])


def triangulate_point(P1, P2, point1, point2):
    design_matrix = build_design_matrix(P1, P2, point1, point2)
    _, _, Vt = jp.linalg.svd(design_matrix)
    return Vt[-1, :3] / Vt[-1, 3]


def triangulate_points(P1, P2, points1, points2):
    triangulate = jax.vmap(triangulate_point, in_axes=(None, None, 0, 0))
    return triangulate(P1, P2, points1, points2)


def recover_pose(essential_matrix, camera_intrinsics, points1, points2):
    """Recovers rotation and translation with the most points in front of
    both cameras (cheirality check), as in Hartley & Zisserman §9.6.3."""
    rotations, translations = decompose_essential_matrix(essential_matrix)
    P1 = paz.pinhole.make_camera_matrix(camera_intrinsics, jp.eye(4))

    def count_points_in_front(rotation, translation):
        pose = paz.pinhole.to_affine_matrix(rotation, translation)
        P2 = paz.pinhole.make_camera_matrix(camera_intrinsics, pose)
        points3D = triangulate_points(P1, P2, points1, points2)
        return jp.sum(points3D[:, 2] > 0.0)

    counts = jax.vmap(count_points_in_front)(rotations, translations)
    best = jp.argmax(counts)
    return rotations[best], translations[best]
