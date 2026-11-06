from functools import partial

import jax
import jax.numpy as jp

TOLERANCE = 1e-6


def compute_mean_value_coordinate(cage_vertices, cage_faces, query):
    vectors = cage_vertices - query
    distances = jp.linalg.norm(vectors, axis=1)
    is_at_vertex = distances < TOLERANCE
    safe_distances = jp.where(is_at_vertex, 1.0, distances)
    units = (vectors / jp.expand_dims(safe_distances, 1))[cage_faces]
    face_args = (units, safe_distances[cage_faces])
    face_weights = compute_face_weights(*face_args)
    weights = sum_face_weights(face_weights, cage_faces, len(cage_vertices))
    vertex_weights = jax.nn.one_hot(jp.argmin(distances), len(cage_vertices))
    return jp.where(jp.any(is_at_vertex), vertex_weights, weights)


def compute_face_weights(units, face_distances):
    angles = compute_corner_angles(units)
    is_on_face = jp.pi - (jp.sum(angles, axis=1) / 2.0) < TOLERANCE
    barycentric = compute_barycentric_weights(angles, face_distances)
    barycentric = jp.where(jp.expand_dims(is_on_face, 1), barycentric, 0.0)
    mean_value = compute_mean_value_weights(units, angles, face_distances)
    return jp.where(jp.any(is_on_face), barycentric, mean_value)


def compute_corner_angles(units):
    angles = []
    for corner in range(3):
        next_units = units[:, (corner + 1) % 3]
        last_units = units[:, (corner + 2) % 3]
        sines = jp.linalg.norm(jp.cross(next_units, last_units), axis=1)
        cosines = jp.sum(next_units * last_units, axis=1)
        angles.append(jp.arctan2(sines, cosines))
    return jp.stack(angles, axis=1)


def compute_barycentric_weights(angles, face_distances):
    weights = []
    for corner in range(3):
        next_distances = face_distances[:, (corner + 1) % 3]
        last_distances = face_distances[:, (corner + 2) % 3]
        area = jp.sin(angles[:, corner]) * next_distances * last_distances
        weights.append(area)
    return jp.stack(weights, axis=1)


def compute_mean_value_weights(units, angles, face_distances):
    cosines, sines = compute_wedge_terms(units, angles)
    is_skipped = jp.any(jp.abs(sines) <= TOLERANCE, axis=1)
    weights = []
    for corner in range(3):
        weight_args = (cosines, sines, angles, face_distances, corner)
        numerator, denominator = compute_weight_terms(*weight_args)
        safe_denominator = jp.where(is_skipped, 1.0, denominator)
        weights.append(jp.where(is_skipped, 0.0, numerator / safe_denominator))
    return jp.stack(weights, axis=1)


def compute_wedge_terms(units, angles):
    half_sum = jp.sum(angles, axis=1) / 2.0
    cosines = []
    for corner in range(3):
        next_sine = jp.sin(angles[:, (corner + 1) % 3])
        last_sine = jp.sin(angles[:, (corner + 2) % 3])
        wedge = jp.sin(half_sum) * jp.sin(half_sum - angles[:, corner])
        denominator = next_sine * last_sine
        is_flat = denominator == 0.0
        safe_denominator = jp.where(is_flat, 1.0, denominator)
        cosine = (2.0 * wedge / safe_denominator) - 1.0
        cosines.append(jp.where(is_flat, 1.0, cosine))
    cosines = jp.stack(cosines, axis=1)
    signs = jp.expand_dims(jp.sign(jp.linalg.det(units)), 1)
    return cosines, signs * jp.sqrt(jp.clip(1.0 - (cosines**2), 0.0, None))


def compute_weight_terms(cosines, sines, angles, face_distances, corner):
    next_corner, last_corner = (corner + 1) % 3, (corner + 2) % 3
    spread = cosines[:, next_corner] * angles[:, last_corner]
    spread = spread + cosines[:, last_corner] * angles[:, next_corner]
    numerator = angles[:, corner] - spread
    denominator = jp.sin(angles[:, next_corner]) * sines[:, last_corner]
    denominator = 2.0 * face_distances[:, corner] * denominator
    return numerator, denominator


def sum_face_weights(face_weights, cage_faces, num_cage_vertices):
    sum_args = (face_weights.ravel(), cage_faces.ravel(), num_cage_vertices)
    return jax.ops.segment_sum(*sum_args)


def compute_mesh_weights(mesh_vertices, cage_vertices, cage_faces, batch_size=256):  # fmt: skip
    compute = partial(compute_mean_value_coordinate, cage_vertices, cage_faces)
    return jax.lax.map(compute, mesh_vertices, batch_size=batch_size)


def control_mesh(mesh_weights, deformed_cage_vertices):
    total_weights = jp.sum(mesh_weights, axis=1, keepdims=True)
    deformed = jp.matmul(mesh_weights, deformed_cage_vertices)
    return deformed / total_weights
