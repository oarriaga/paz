import jax.numpy as jp

from paz.graphics.geometry import (
    replace_misses,
    compute_points3D,
    compute_quadratic_depths,
)

from paz.graphics.shapes.quadratics import (
    solve_quadratic_depth,
    get_ring_quadratic_coefficients,
)


def intersect_infinite_ring(ray_origins, ray_directions):
    a, b, c = get_ring_quadratic_coefficients(ray_origins, ray_directions)
    valid_mask, side_depth_A, side_depth_B = solve_quadratic_depth(a, b, c)
    return valid_mask, side_depth_A, side_depth_B


def bound_infinite_ring(depth, ray_origins, ray_directions, minimum, maximum):
    points3D = compute_points3D(ray_origins, ray_directions, depth)
    points3D_y = points3D[:, 1:2]
    above_min = minimum < points3D_y
    below_max = points3D_y < maximum
    finite_ring_mask = jp.logical_and(above_min, below_max)
    return finite_ring_mask


def intersect_canonical_ring(ray_origins, ray_directions, minimum, maximum):
    intersections = intersect_infinite_ring(ray_origins, ray_directions)
    valid_mask, infinite_ring_depth_A, infinite_ring_depth_B = intersections
    args = (ray_origins, ray_directions, minimum, maximum)
    finite_ring_mask_A = bound_infinite_ring(infinite_ring_depth_A, *args)
    finite_ring_mask_B = bound_infinite_ring(infinite_ring_depth_B, *args)
    finite_ring_mask_A = jp.logical_and(valid_mask, finite_ring_mask_A)
    finite_ring_mask_B = jp.logical_and(valid_mask, finite_ring_mask_B)
    ring_depth_A = replace_misses(infinite_ring_depth_A, finite_ring_mask_A)
    ring_depth_B = replace_misses(infinite_ring_depth_B, finite_ring_mask_B)
    sorted_depths = jp.hstack([ring_depth_A, ring_depth_B])  # TODO SORT

    depths = compute_quadratic_depths(ring_depth_A, ring_depth_B)
    ring_mask = jp.logical_or(finite_ring_mask_A, finite_ring_mask_B)
    depths = replace_misses(depths, ring_mask)
    ring_mask = jp.squeeze(ring_mask, -1)

    return ring_mask, sorted_depths, depths


def build_canonical_ring_normals(points):
    x_points, y_points, z_points = jp.split(points, 3, 1)
    zeros = jp.zeros((len(points), 1))
    ring_normals = jp.hstack([x_points, zeros, z_points])
    return ring_normals
