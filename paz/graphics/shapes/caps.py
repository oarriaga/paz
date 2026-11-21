import jax.numpy as jp

from paz.graphics.constants import EPSILON


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
    return (y_translation - y_origins) / y_directions
