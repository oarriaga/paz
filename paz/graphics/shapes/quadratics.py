import jax.numpy as jp
import paz

from paz.graphics.constants import EPSILON
from paz.graphics.geometry import (
    compute_points3D,
    replace_misses,
    compute_quadratic_depths,
)
from paz.graphics.shapes.caps import (
    intersect_cone_caps,
    intersect_cylinder_caps,
)


def get_ring_quadratic_coefficients(ray_origins, ray_directions):
    directions = jp.split(ray_directions, 3, 1)
    x_directions, y_directions, z_directions = directions
    x_origins, y_origins, z_origins = jp.split(ray_origins, 3, 1)
    a = x_directions**2 + z_directions**2
    b = 2 * ((x_origins * x_directions) + (z_origins * z_directions))
    c = x_origins**2 + z_origins**2 - 1
    return a, b, c


def get_cone_quadratic_coefficients(ray_origins, ray_directions):
    x_directions, y_directions, z_directions = jp.split(ray_directions, 3, 1)
    x_origins, y_origins, z_origins = jp.split(ray_origins, 3, 1)
    a = x_directions**2 - y_directions**2 + z_directions**2
    b = 2 * (
        (x_origins * x_directions)
        - (y_origins * y_directions)
        + (z_origins * z_directions)
    )
    c = x_origins**2 - y_origins**2 + z_origins**2
    return a, b, c


def compute_bounding_mask(points3D, minimum, maximum):
    points3D_y = points3D[:, 1:2]
    above_min = minimum < points3D_y
    below_max = points3D_y < maximum
    bounding_mask = jp.logical_and(above_min, below_max)
    return bounding_mask


def bound_depth(depth, origins, directions, valid_mask, minimum, maximum):
    points3D = compute_points3D(origins, directions, depth)
    bounding_mask = compute_bounding_mask(points3D, minimum, maximum)
    # bounding_mask = jp.logical_and(valid_mask, bounding_mask)
    valid_depth_mask = jp.logical_and(valid_mask, depth > EPSILON)
    bounding_mask = jp.logical_and(valid_depth_mask, bounding_mask)
    bounded_depth = replace_misses(depth, bounding_mask)
    return bounding_mask, bounded_depth


def solve_quadratic_depth(a, b, c):
    depth_A, depth_B, valid_mask = paz.algebra.solve_quadratic(a, b, c)
    positive_depth_side = jp.logical_or(depth_A >= 0, depth_B >= 0)
    valid_mask = jp.logical_and(valid_mask, positive_depth_side)
    return valid_mask, depth_A, depth_B


def intersect_quadratic(a, b, c, origins, directions, minimum, maximum):
    valid_mask, depth_A, depth_B = solve_quadratic_depth(a, b, c)
    args = (origins, directions, valid_mask, minimum, maximum)
    bounding_mask_A, bounded_depth_A = bound_depth(depth_A, *args)
    bounding_mask_B, bounded_depth_B = bound_depth(depth_B, *args)
    mask = jp.logical_or(bounding_mask_A, bounding_mask_B)
    depths = compute_quadratic_depths(bounded_depth_A, bounded_depth_B)
    depths = replace_misses(depths, mask)
    sorted_depths = jp.vstack([bounded_depth_A[:, 0], bounded_depth_B[:, 0]])
    return jp.squeeze(mask, -1), sorted_depths, depths


def intersect_caped_quadratic(body_intersections, caps_intersections):
    body_mask, body_depths, body_depth = body_intersections
    caps_mask, caps_depths, caps_depth = caps_intersections
    body_mask = jp.expand_dims(body_mask, axis=1)
    caps_mask = jp.expand_dims(caps_mask, axis=1)
    hit_mask = jp.logical_or(body_mask, caps_mask)
    depths = jp.concatenate([body_depth, caps_depth], axis=1)
    depth = jp.min(depths, axis=1, keepdims=True)
    depth = replace_misses(depth, hit_mask)

    hit_mask = jp.squeeze(hit_mask, axis=1)
    depths = jp.vstack([body_depths, caps_depths])
    paz.graphics.shapes.pad_depths(depths, hit_mask, 0)
    return hit_mask, depths, depth


def intersect_canonical_cylinder(origins, directions):
    minimum = -1.0
    maximum = 1.0
    a, b, c = get_ring_quadratic_coefficients(origins, directions)
    args = (origins, directions, minimum, maximum)
    body_intersections = intersect_quadratic(a, b, c, *args)
    caps_intersections = intersect_cylinder_caps(*args)
    return intersect_caped_quadratic(body_intersections, caps_intersections)


# def intersect_canonical_cylinder(origins, directions):
#     minimum, maximum = -1.0, 1.0
#     a, b, c = get_ring_quadratic_coefficients(origins, directions)
#     args = (origins, directions, minimum, maximum)
#     body_intersections = intersect_quadratic(a, b, c, *args)
#     body_hit_mask, body_depths, body_depth = body_intersections
#     caps_intersections = intersect_cylinder_caps(*args)
#     caps_hit_mask, caps_depths, caps_depth = caps_intersections
#     hit_mask = jp.logical_or(body_hit_mask, caps_hit_mask)
#     depths = jp.concatenate([body_depths, caps_depths], axis=0)
#     depth = jp.expand_dims(jp.min(depths, axis=0), axis=1)
#     return hit_mask, depths, depth


def intersect_canonical_cone(origins, directions):
    minimum, maximum = -1.0, 0.0
    a, b, c = get_cone_quadratic_coefficients(origins, directions)
    args = (origins, directions, minimum, maximum)
    body_intersections = intersect_quadratic(a, b, c, *args)
    caps_intersections = intersect_cone_caps(*args)
    return intersect_caped_quadratic(body_intersections, caps_intersections)
