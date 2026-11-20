import jax
import jax.numpy as jp

import paz
from paz.graphics.shapes.caps import build_lower_cap_normals
from paz.graphics.shapes.caps import build_upper_cap_normals
from paz.graphics.shapes.cone import build_shell_normals
from paz.graphics.shapes.cube import check_axis
from paz.graphics.shapes.cube import compute_canonical_normals_cube
from paz.graphics.shapes.plane import compute_canonical_normals_plane
from paz.graphics.shapes.quadratics import get_cone_quadratic_coefficients
from paz.graphics.shapes.quadratics import get_ring_quadratic_coefficients
from paz.graphics.shapes.ring import build_canonical_ring_normals


def compute_interval(shape, origins, directions):
    """Depths at which each ray enters and leaves the inside of a shape.

    A ray that misses gets the empty interval `(inf, -inf)`. Depths behind
    the origin are kept, so a ray starting inside the shape is still inside.
    """
    world_to_shape = jp.linalg.inv(shape.transform)
    transform_args = world_to_shape, origins, directions
    rays = paz.graphics.geometry.transform_rays(*transform_args)
    cases = [compute_sphere_interval, compute_cube_interval]
    cases += [compute_cylinder_interval, compute_cone_interval]
    cases += [compute_plane_interval]
    return jax.lax.switch(shape.type, cases, *rays)


def compute_normals(shape, points, eyes):
    """Normals of a shape at surface `points`, turned to face `eyes`."""
    world_to_shape = jp.linalg.inv(shape.transform)
    shape_points = paz.algebra.transform_points(world_to_shape, points)
    cases = [compute_sphere_normals, compute_canonical_normals_cube]
    cases += [compute_cylinder_normals, compute_cone_normals]
    cases += [compute_canonical_normals_plane]
    normals = jax.lax.switch(shape.type, cases, shape_points)
    normals = paz.algebra.transform_points(world_to_shape.T, normals)
    normals = paz.algebra.normalize(normals)
    return paz.graphics.shapes.invert_inside_normals(eyes, normals)


def compute_sphere_interval(origins, directions):
    a = paz.algebra.dot(directions, directions)
    b = 2.0 * paz.algebra.dot(directions, origins)
    c = paz.algebra.dot(origins, origins) - 1.0
    depths_A, depths_B, is_valid = paz.algebra.solve_quadratic(a, b, c)
    return build_interval(depths_A, depths_B, is_valid)


def compute_cube_interval(origins, directions):
    axis_enters, axis_exits = check_axis(origins, directions)
    enters = jp.max(axis_enters, axis=1)
    exits = jp.min(axis_exits, axis=1)
    return build_interval(enters, exits, True)


def compute_cylinder_interval(origins, directions):
    a, b, c = get_ring_quadratic_coefficients(origins, directions)
    side_enters, side_exits = solve_ring_interval(a[:, 0], b[:, 0], c[:, 0])
    cap_enters, cap_exits = check_axis(origins[:, 1], directions[:, 1])
    enters = jp.maximum(side_enters, cap_enters)
    exits = jp.minimum(side_exits, cap_exits)
    return build_interval(enters, exits, True)


def compute_cone_interval(origins, directions):
    side_enters, side_exits = solve_cone_interval(origins, directions)
    # The cone spans y in [-1, 0]; `check_axis` expects [-1, 1].
    y_origins, y_directions = origins[:, 1], directions[:, 1]
    cap_args = 2.0 * y_origins + 1.0, 2.0 * y_directions
    cap_enters, cap_exits = check_axis(*cap_args)
    enters = jp.maximum(side_enters, cap_enters)
    exits = jp.minimum(side_exits, cap_exits)
    return build_interval(enters, exits, True)


def compute_plane_interval(origins, directions):
    """Treats the plane as the half-space below it, `y < 0`."""
    y_directions = directions[:, 1]
    sign = jp.where(y_directions >= 0.0, 1e-7, -1e-7)
    is_safe = jp.abs(y_directions) > 1e-7
    safe_directions = jp.where(is_safe, y_directions, sign)
    depths = -origins[:, 1] / safe_directions
    is_descending = safe_directions < 0.0
    enters = jp.where(is_descending, depths, -jp.inf)
    exits = jp.where(is_descending, jp.inf, depths)
    return enters, exits


def build_interval(enters, exits, is_valid):
    is_valid = is_valid & (exits > enters)
    enters = jp.where(is_valid, enters, jp.inf)
    exits = jp.where(is_valid, exits, -jp.inf)
    return enters, exits


def solve_ring_interval(a, b, c):
    # A ray along the axis never crosses the side: it is inside or outside.
    is_axial = a < 1e-12
    safe_a = jp.where(is_axial, 1.0, a)
    roots = paz.algebra.solve_quadratic(safe_a, b, c)
    depths_A, depths_B, is_valid = roots
    enters = jp.where(is_axial, -jp.inf, depths_A)
    exits = jp.where(is_axial, jp.inf, depths_B)
    is_valid = jp.where(is_axial, c < 0.0, is_valid)
    return build_interval(enters, exits, is_valid)


def solve_cone_interval(origins, directions):
    a, b, c = get_cone_quadratic_coefficients(origins, directions)
    a, b, c = a[:, 0], b[:, 0], c[:, 0]
    safe_a = jp.where(jp.abs(a) > 1e-7, a, 1e-7)
    roots = paz.algebra.solve_quadratic(safe_a, b, c)
    depths_A, depths_B, is_valid = roots
    lower = jp.minimum(depths_A, depths_B)
    upper = jp.maximum(depths_A, depths_B)
    steep_args = lower, upper, directions[:, 1] < 0.0
    steep_enters, steep_exits = build_steep_cone_interval(*steep_args)
    is_steep = safe_a < 0.0
    enters = jp.where(is_steep, steep_enters, lower)
    exits = jp.where(is_steep, steep_exits, upper)
    return build_interval(enters, exits, is_valid)


# A ray steeper than the cone is inside the double cone outside its two
# roots. Only the half that heads down the lower nappe belongs to the cone.
def build_steep_cone_interval(lower, upper, is_descending):
    enters = jp.where(is_descending, upper, -jp.inf)
    exits = jp.where(is_descending, jp.inf, lower)
    return enters, exits


def compute_sphere_normals(points):
    return points


def compute_cylinder_normals(points):
    x, y, z = jp.split(points, 3, axis=1)
    is_cap = jp.abs(y) > jp.sqrt(x**2 + z**2)
    upper_cap_normals = build_upper_cap_normals(points)
    lower_cap_normals = build_lower_cap_normals(points)
    cap_normals = jp.where(y > 0.0, upper_cap_normals, lower_cap_normals)
    return jp.where(is_cap, cap_normals, build_canonical_ring_normals(points))


def compute_cone_normals(points):
    x, y, z = jp.split(points, 3, axis=1)
    radius = jp.sqrt(x**2 + z**2)
    is_cap = jp.abs(y + 1.0) < jp.abs(radius + y)
    cap_normals = build_lower_cap_normals(points)
    return jp.where(is_cap, cap_normals, build_shell_normals(points))
