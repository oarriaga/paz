import jax.numpy as jp

from paz.backend.graphics.shapes.quadratics import intersect_canonical_cone
from paz.backend.graphics.shapes.caps import (
    build_lower_cap_normals,
    build_upper_cap_normals,
    compute_cap_masks,
    compute_inner_cone_cap_mask,
)


def build_shell_normals(points):
    x_points, y_points, z_points = jp.split(points, 3, 1)
    distances = jp.sqrt(x_points**2 + z_points**2)
    y_normals = jp.where(y_points > 0, -distances, distances)
    shape_normals = jp.hstack([x_points, y_normals, z_points])
    return shape_normals


def compute_canonical_normals_cone(points):
    minimum, maximum = -1.0, 0.0
    inner_cap_mask = compute_inner_cone_cap_mask(points)
    lower_cap_mask, upper_cap_mask = compute_cap_masks(
        points, inner_cap_mask, minimum, maximum
    )
    upper_cap_normals = build_upper_cap_normals(points)
    lower_cap_normals = build_lower_cap_normals(points)
    shell_normals = build_shell_normals(points)

    normals = jp.zeros((len(points), 3))
    normals = jp.where(lower_cap_mask, lower_cap_normals, normals)
    normals = jp.where(upper_cap_mask, upper_cap_normals, normals)
    cap_mask = jp.logical_or(upper_cap_mask, lower_cap_mask)
    normals = jp.where(cap_mask, normals, shell_normals)
    return normals
