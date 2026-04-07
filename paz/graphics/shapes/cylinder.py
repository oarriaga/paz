import jax.numpy as jp

from paz.graphics.shapes.ring import build_canonical_ring_normals
from paz.graphics.shapes.quadratics import intersect_canonical_cylinder
from paz.graphics.shapes.caps import (
    build_upper_cap_normals,
    build_lower_cap_normals,
)


def compute_canonical_normals_cylinder(points, sorted_depths):
    body_depth = jp.min(sorted_depths[:2], axis=0)
    lower_cap_depth = sorted_depths[2]
    upper_cap_depth = sorted_depths[3]
    depths = jp.stack([body_depth, lower_cap_depth, upper_cap_depth])
    closest_depth_arg = jp.expand_dims(jp.argmin(depths, axis=0), axis=1)
    upper_cap_normals = build_upper_cap_normals(points)
    lower_cap_normals = build_lower_cap_normals(points)
    ring_normals = build_canonical_ring_normals(points)
    lower_cap_mask = closest_depth_arg == 1  # CYLINDER_LOWER_CAP
    upper_cap_mask = closest_depth_arg == 2  # CYLINDER_UPPER_CAP
    normals = jp.zeros((len(points), 3))
    normals = jp.where(lower_cap_mask, lower_cap_normals, normals)
    normals = jp.where(upper_cap_mask, upper_cap_normals, normals)
    cap_mask = jp.logical_or(upper_cap_mask, lower_cap_mask)
    normals = jp.where(cap_mask, normals, ring_normals)
    return normals
