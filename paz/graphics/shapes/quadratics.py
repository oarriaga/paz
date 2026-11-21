import jax.numpy as jp
import paz

from paz.graphics.constants import FARAWAY
from paz.graphics.geometry import is_ahead, replace_misses
from paz.graphics.shapes.caps import intersect_infinite_plane


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


def solve_quadratic_depth(a, b, c):
    depth_A, depth_B, valid_mask = paz.algebra.solve_quadratic(a, b, c)
    return valid_mask, depth_A, depth_B


def intersect_canonical_cylinder(origins, directions):
    coefficients = get_ring_quadratic_coefficients(origins, directions)
    return intersect_capped(*coefficients, origins, directions, -1.0, 1.0)


def intersect_canonical_cone(origins, directions):
    coefficients = get_cone_quadratic_coefficients(origins, directions)
    return intersect_capped(*coefficients, origins, directions, -1.0, 0.0)


def intersect_capped(a, b, c, origins, directions, minimum, maximum):
    has_sides, side_A, side_B = solve_quadratic_depth(a, b, c)
    sides = jp.hstack([side_A, side_B])
    has_sides = has_sides & jp.all(jp.isfinite(sides), axis=1, keepdims=True)
    lower_cap = intersect_infinite_plane(origins, directions, minimum)
    upper_cap = intersect_infinite_plane(origins, directions, maximum)
    caps = jp.hstack([lower_cap, upper_cap])
    is_side = has_sides & is_between(sides, lower_cap, upper_cap)
    is_cap = is_inside_quadratic(a, c, has_sides, sides, caps)
    is_root = jp.hstack([is_side, is_cap])
    return select_roots(is_root, jp.hstack([sides, caps]))


# A side root is real between the caps and a cap root is real inside the
# quadratic. Both compare the same depths, one including the tie the other
# excludes, so a ray crossing a rim gets an even number of roots.
def is_between(depths, depth_A, depth_B):
    lower = jp.minimum(depth_A, depth_B)
    upper = jp.maximum(depth_A, depth_B)
    return (lower <= depths) & (depths < upper)


# Without side roots a ray is inside everywhere or nowhere, as at its origin.
def is_inside_quadratic(a, c, has_sides, sides, depths):
    near = jp.min(sides, axis=1, keepdims=True)
    far = jp.max(sides, axis=1, keepdims=True)
    is_between_sides = (near < depths) & (depths <= far)
    is_inside = is_between_sides == (a > 0.0)
    return jp.where(has_sides, is_inside, c <= 0.0)


def select_roots(is_root, roots):
    roots = replace_misses(roots, is_root)
    ahead_roots = replace_misses(roots, is_ahead(roots))
    depth = jp.min(ahead_roots, axis=1, keepdims=True)
    return depth[:, 0] < FARAWAY, roots.T, depth
