from paz.backend import algebra

from functools import partial
import jax.numpy as jp
from jax import vmap


def scale_intrinsics(camera_intrinsics, scale):
    focal_length_x = scale * camera_intrinsics[0, 0]
    focal_length_y = scale * camera_intrinsics[1, 1]
    center_x = scale * camera_intrinsics[0, 2]
    center_y = scale * camera_intrinsics[1, 2]
    scaled_intrinsics = jp.array([[focal_length_x, 0.0, center_x],
                                  [0.0, focal_length_y, center_y],
                                  [0.0, 0.0, 1.0]])
    return scaled_intrinsics


def project_2D_to_3D(camera_intrinsics, point2D, depth):
    u, v = point2D
    pixel = jp.array([u, v, 1.0])
    point3D = jp.linalg.inv(camera_intrinsics) @ pixel * -depth
    return point3D


def project_3D_to_2D(camera_matrix, point3D):
    homogenous_point3D = algebra.add_one(point3D)
    homogenous_point2D = jp.matmul(camera_matrix, homogenous_point3D)
    point2D = algebra.dehomogenize_coordinates(homogenous_point2D)
    return point2D


def build_cube_corners(min_extents, max_extents):
    x_min, y_min, z_min = min_extents
    x_max, y_max, z_max = max_extents
    return jp.array([[x_min, y_max, z_min],
                     [x_max, y_max, z_min],
                     [x_max, y_min, z_min],
                     [x_min, y_min, z_min],
                     [x_min, y_max, z_max],
                     [x_max, y_max, z_max],
                     [x_max, y_min, z_max],
                     [x_min, y_min, z_max]])


def compute_AABB(vertices):
    min_extents = jp.min(vertices, axis=0)
    max_extents = jp.max(vertices, axis=0)
    return build_cube_corners(min_extents, max_extents)


def to_affine_matrix(rotation_matrix, translation):
    translation = translation.reshape(3, 1)
    affine_top = jp.concatenate([rotation_matrix, translation], axis=1)
    affine_row = jp.array([[0.0, 0.0, 0.0, 1.0]])
    affine_matrix = jp.concatenate([affine_top, affine_row], axis=0)
    return affine_matrix


def compute_OBB(vertices_world):
    centroid_world = jp.mean(vertices_world, axis=0)
    centered_vertices_world = vertices_world - centroid_world
    centered_vertices_world_xy = centered_vertices_world[:, [0, 2]]
    u, s, vh = jp.linalg.svd(centered_vertices_world_xy)
    principal_axes = vh.T
    rx, ry, ux, uz = principal_axes.flatten()
    principal_axes = jp.array([
        [rx, 0.0, ry],
        [0., 1.0, 0],
        [ux, 0.0, uz]])
    centered_vertices_alpha = (principal_axes @ centered_vertices_world.T).T
    min_extents_alpha = jp.min(centered_vertices_alpha, axis=0)
    max_extents_alpha = jp.max(centered_vertices_alpha, axis=0)
    corners_world = build_cube_corners(min_extents_alpha, max_extents_alpha)
    world_to_alpha = to_affine_matrix(principal_axes, centroid_world)
    return algebra.transform_points(world_to_alpha, corners_world), world_to_alpha


def project_box_and_vertices(intrinsics, world_to_camera, vertices_world):
    points3D, world_to_alpha = compute_OBB(vertices_world)
    camera_matrix = intrinsics @ world_to_camera
    points2D = vmap(partial(project_3D_to_2D, camera_matrix))(points3D)
    vertices2D = vmap(partial(project_3D_to_2D, camera_matrix))(vertices_world)
    return points2D.astype(int), vertices2D.astype(int)
