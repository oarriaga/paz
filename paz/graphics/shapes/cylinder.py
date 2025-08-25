import jax.numpy as jp

from paz.graphics.shapes.ring import build_canonical_ring_normals
from paz.graphics.shapes.quadratics import intersect_canonical_cylinder
from paz.graphics.shapes.caps import (
    build_upper_cap_normals,
    build_lower_cap_normals,
    compute_cap_masks,
    compute_inner_cylinder_cap_mask,
)


def compute_canonical_normals_cylinder(points):
    minimum, maximum = -1.0, 1.0
    inner_cap_mask = compute_inner_cylinder_cap_mask(points)
    lower_cap_mask, upper_cap_mask = compute_cap_masks(
        points, inner_cap_mask, minimum, maximum
    )
    upper_cap_normals = build_upper_cap_normals(points)
    lower_cap_normals = build_lower_cap_normals(points)
    ring_normals = build_canonical_ring_normals(points)
    normals = jp.zeros((len(points), 3))
    normals = jp.where(lower_cap_mask, lower_cap_normals, normals)
    normals = jp.where(upper_cap_mask, upper_cap_normals, normals)
    cap_mask = jp.logical_or(upper_cap_mask, lower_cap_mask)
    normals = jp.where(cap_mask, normals, ring_normals)
    return normals
