import jax.numpy as jp

import paz


def compute_position(ray_origins, ray_directions, depths):
    ray_directions = jp.expand_dims(ray_directions, axis=0)
    ray_origins = jp.expand_dims(ray_origins, axis=0)
    positions = ray_origins + (depths * ray_directions)
    return positions


def extract_points(vertices, faces):
    points_A = vertices[faces[:, 0]]
    points_B = vertices[faces[:, 1]]
    points_C = vertices[faces[:, 2]]
    return points_A, points_B, points_C


def build_edges(vertices, faces):
    points_A, points_B, points_C = extract_points(vertices, faces)
    edges_AB = points_B - points_A
    edges_AC = points_C - points_A
    edges_AC = jp.expand_dims(edges_AC, axis=1)
    edges_AB = jp.expand_dims(edges_AB, axis=1)
    points_A = jp.expand_dims(points_A, axis=1)
    return edges_AC, edges_AB, points_A


def compute_canonical_normals(vertices, faces, shape_points):
    points_A, points_B, points_C = extract_points(vertices, faces)
    edges_AB = points_B - points_A
    edges_AC = points_C - points_A
    normals = jp.cross(edges_AB, edges_AC)
    normals = paz.algebra.normalize(normals)
    num_rays = shape_points.shape[1]
    normals = jp.expand_dims(normals, 1)
    normals = jp.repeat(normals, num_rays, axis=1)
    return normals


def transform_points(affine_transform, points):
    ones = jp.ones((*points.shape[:2], 1))
    points = jp.concatenate([points, ones], axis=-1)
    points = jp.swapaxes(points, 1, 2)
    points = jp.matmul(affine_transform, points)
    points = jp.swapaxes(points, 2, 1)
    return points[:, :, :3]


def compute_normals(vertices, faces, transform, world_points):
    inverse = jp.linalg.inv(transform)
    shape_points = transform_points(inverse, world_points)
    normals = compute_canonical_normals(vertices, faces, shape_points)
    normals = transform_points(inverse.T, normals)
    normals = paz.algebra.normalize(normals)
    return normals


def compute_normals_for_hits(vertices, faces, transform, face_indices):
    hit_faces = faces[face_indices]
    A = vertices[hit_faces[:, 0]]
    B = vertices[hit_faces[:, 1]]
    C = vertices[hit_faces[:, 2]]
    normals = jp.cross(B - A, C - A)
    normals = paz.algebra.normalize(normals)
    inverse = jp.linalg.inv(transform)
    normals = normals @ inverse[:3, :3]
    return paz.algebra.normalize(normals)
