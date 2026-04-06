from functools import partial

import jax
import jax.numpy as jp


def compute_mean_value_coordinate(cage_vertices, cage_faces, query):
    cage_vectors = cage_vertices - query
    cage_distances = jp.linalg.norm(cage_vectors, axis=1, keepdims=True)
    unit_vectors = cage_vectors / (cage_distances + 1e-5)
    unit_triangles = unit_vectors[cage_faces]
    u1 = unit_triangles[:, 0]
    u2 = unit_triangles[:, 1]
    u3 = unit_triangles[:, 2]
    l1 = jp.linalg.norm(u2 - u3, axis=1)
    l2 = jp.linalg.norm(u3 - u1, axis=1)
    l3 = jp.linalg.norm(u1 - u2, axis=1)
    theta_1 = 2.0 * jp.arcsin(jp.clip(l1 / 2.0, -1.0, 1.0))
    theta_2 = 2.0 * jp.arcsin(jp.clip(l2 / 2.0, -1.0, 1.0))
    theta_3 = 2.0 * jp.arcsin(jp.clip(l3 / 2.0, -1.0, 1.0))
    h = (theta_1 + theta_2 + theta_3) / 2.0
    c1_num = 2.0 * jp.sin(h) * jp.sin(h - theta_1)
    c2_num = 2.0 * jp.sin(h) * jp.sin(h - theta_2)
    c3_num = 2.0 * jp.sin(h) * jp.sin(h - theta_3)
    c1_den = jp.sin(theta_2) * jp.sin(theta_3) + 1e-8
    c2_den = jp.sin(theta_3) * jp.sin(theta_1) + 1e-8
    c3_den = jp.sin(theta_1) * jp.sin(theta_2) + 1e-8
    c1 = c1_num / c1_den - 1.0
    c2 = c2_num / c2_den - 1.0
    c3 = c3_num / c3_den - 1.0
    signs = jp.sign(jp.linalg.det(unit_triangles))
    s1 = signs * jp.sqrt(jp.clip(1 - c1**2, 0.0, None))
    s2 = signs * jp.sqrt(jp.clip(1 - c2**2, 0.0, None))
    s3 = signs * jp.sqrt(jp.clip(1 - c3**2, 0.0, None))
    w1_num = theta_1 - c2 * theta_3 - c3 * theta_2
    w2_num = theta_2 - c3 * theta_1 - c1 * theta_3
    w3_num = theta_3 - c1 * theta_2 - c2 * theta_1
    face_distances = cage_distances[cage_faces]
    d1 = face_distances[:, 0, 0]
    d2 = face_distances[:, 1, 0]
    d3 = face_distances[:, 2, 0]
    w1_den = 2.0 * d1 * jp.sin(theta_2) * s3 + 1e-8
    w2_den = 2.0 * d2 * jp.sin(theta_3) * s1 + 1e-8
    w3_den = 2.0 * d3 * jp.sin(theta_1) * s2 + 1e-8
    w1 = w1_num / w1_den
    w2 = w2_num / w2_den
    w3 = w3_num / w3_den
    weights_per_face = jp.stack((w1, w2, w3), axis=1)
    weights = []
    for arg in range(len(cage_vertices)):
        cage_vertex_mask = cage_faces == arg
        cage_vertex_weight = jp.sum(weights_per_face[cage_vertex_mask])
        weights.append(cage_vertex_weight)
    return jp.array(weights)


def compute_mesh_weights(mesh_vertices, cage_vertices, cage_faces):
    cage_args = (cage_vertices, cage_faces)
    compute_mvc = jax.vmap(partial(compute_mean_value_coordinate, *cage_args))
    return compute_mvc(mesh_vertices)


def control_mesh(mesh_weights, deformed_cage_vertices):
    total_weights = jp.sum(mesh_weights, axis=1, keepdims=True)
    deformed = jp.matmul(mesh_weights, deformed_cage_vertices)
    return deformed / total_weights
