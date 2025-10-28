import jax.numpy as jp

from paz.graphics.constants import EPSILON, FARAWAY
from paz.graphics.geometry import (
    compute_quadratic_depths,
    compute_points3D,
    replace_misses,
)


def compute_inner_cylinder_cap_mask(points):
    x = points[:, 0:1]
    z = points[:, 2:3]
    cap_mask = (x**2 + z**2) <= 1.0
    return cap_mask


def compute_inner_cone_cap_mask(points):
    x, y, z = jp.split(points, 3, axis=1)
    cap_mask = (x**2 + z**2) <= y**2
    return cap_mask


def compute_lower_cap_mask(points, inner_cap_mask, minimum, epsilon=EPSILON):
    y_points = points[:, 1:2]
    lower_cap_mask = y_points <= minimum + epsilon
    lower_cap_mask = jp.logical_and(inner_cap_mask, lower_cap_mask)
    return lower_cap_mask


def compute_upper_cap_mask(points, inner_cap_mask, maximum, epsilon=EPSILON):
    y_points = points[:, 1:2]
    upper_cap_mask = y_points >= maximum - epsilon
    upper_cap_mask = jp.logical_and(inner_cap_mask, upper_cap_mask)
    return upper_cap_mask


def compute_cap_masks(points, inner_cap_mask, minimum, maximum):
    lower_cap_mask = compute_lower_cap_mask(points, inner_cap_mask, minimum)
    upper_cap_mask = compute_upper_cap_mask(points, inner_cap_mask, maximum)
    return lower_cap_mask, upper_cap_mask


def build_upper_cap_normals(points):
    num_rays = len(points)
    zeros, ones = jp.zeros((num_rays, 1)), jp.ones((num_rays, 1))
    upper_cap_normals = jp.hstack([zeros, ones, zeros])
    return upper_cap_normals


def build_lower_cap_normals(points):
    num_rays = len(points)
    zeros, ones = jp.zeros((num_rays, 1)), jp.ones((num_rays, 1))
    lower_cap_normals = jp.hstack([zeros, -ones, zeros])
    return lower_cap_normals


def build_canonical_caps_normals(points, inner_cap_mask, minimum, maximum):
    lower_cap_normals = build_lower_cap_normals(points)
    upper_cap_normals = build_upper_cap_normals(points)

    caps_masks = compute_cap_masks(points, inner_cap_mask, minimum, maximum)
    lower_cap_mask, upper_cap_mask = caps_masks
    # TODO check that order does not affect computation
    cap_normals = jp.zeros((len(points), 3))
    cap_normals = jp.where(upper_cap_mask, upper_cap_normals, cap_normals)
    cap_normals = jp.where(lower_cap_mask, lower_cap_normals, cap_normals)
    return cap_normals


def intersect_infinite_plane(ray_origins, ray_directions, y_translation=0.0):
    y_origins, y_directions = ray_origins[:, 1:2], ray_directions[:, 1:2]
    depths = (y_translation - y_origins) / y_directions
    # TODO understand why is needed to fix cylinder shadow error.
    valid_mask = jp.logical_and((depths > EPSILON), (depths < FARAWAY))
    depths = jp.where(valid_mask, depths, FARAWAY)
    # -------------------------------------------------------------------------
    return depths


def intersect_plane(origins, directions, y_translation):
    depths = intersect_infinite_plane(origins, directions, y_translation)
    points3D = compute_points3D(origins, directions, depths)
    return depths, points3D


def intersect_canonical_caps(caps_mask, lower_cap_depth, upper_cap_depth):
    # TODO missing sort
    caps_depths = jp.vstack([lower_cap_depth[:, 0], upper_cap_depth[:, 0]])
    caps_depth = compute_quadratic_depths(lower_cap_depth, upper_cap_depth)

    caps_depth = replace_misses(caps_depth, caps_mask)
    caps_mask = jp.squeeze(caps_mask, axis=1)
    return caps_mask, caps_depths, caps_depth


def intersect_cylinder_caps(origins, directions, minimum, maximum):
    lower_depth, lower_points3D = intersect_plane(origins, directions, minimum)
    upper_depth, upper_points3D = intersect_plane(origins, directions, maximum)
    lower_cap_mask = compute_inner_cylinder_cap_mask(lower_points3D)
    upper_cap_mask = compute_inner_cylinder_cap_mask(upper_points3D)
    lower_depth = replace_misses(lower_depth, lower_cap_mask)
    upper_depth = replace_misses(upper_depth, upper_cap_mask)
    caps_mask = jp.logical_or(upper_cap_mask, lower_cap_mask)
    return intersect_canonical_caps(caps_mask, lower_depth, upper_depth)


def intersect_cone_caps(origins, directions, minimum, maximum):
    lower_depth, lower_points3D = intersect_plane(origins, directions, minimum)
    # upper_depth, upper_points3D = intersect_plane(origins, directions, maximum)
    lower_cap_mask = compute_inner_cone_cap_mask(lower_points3D)
    valid_mask = jp.logical_and(
        (lower_depth > EPSILON), (lower_depth < FARAWAY)
    )
    lower_cap_mask = jp.logical_and(lower_cap_mask, valid_mask)
    # upper_cap_mask = compute_inner_cone_cap_mask(upper_points3D)
    lower_depth = replace_misses(lower_depth, lower_cap_mask)
    # upper_depth = replace_misses(upper_depth, upper_cap_mask)
    # caps_mask = jp.logical_or(upper_cap_mask, lower_cap_mask)
    return intersect_canonical_caps(
        lower_cap_mask, lower_depth, jp.full_like(lower_depth, FARAWAY)
    )
