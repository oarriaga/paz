import paz
import jax.numpy as jp

from paz.graphics.shapes.quadratics import intersect_canonical_cone
from paz.graphics.shapes.caps import (
    build_lower_cap_normals,
    build_upper_cap_normals,
)


def build_shell_normals(points):
    x_points, y_points, z_points = jp.split(points, 3, 1)
    distances = jp.sqrt(x_points**2 + z_points**2)
    y_normals = jp.where(y_points > 0, -distances, distances)
    shape_normals = jp.hstack([x_points, y_normals, z_points])
    return paz.algebra.normalize(shape_normals)


def compute_canonical_normals_cone(points, sorted_depths):
    body_depth = jp.min(sorted_depths[:2], axis=0)
    lower_cap_depth = sorted_depths[2]
    upper_tip_depth = sorted_depths[3]
    feature_depths = jp.stack([body_depth, lower_cap_depth, upper_tip_depth])
    hit_features = jp.expand_dims(jp.argmin(feature_depths, axis=0), axis=1)
    upper_cap_normals = build_upper_cap_normals(points)
    lower_cap_normals = build_lower_cap_normals(points)
    shell_normals = build_shell_normals(points)
    lower_cap_mask = hit_features == 1  # LOWER_CAP
    upper_cap_mask = hit_features == 2  # UPPER_TIP

    normals = jp.zeros((len(points), 3))
    normals = jp.where(lower_cap_mask, lower_cap_normals, normals)
    normals = jp.where(upper_cap_mask, upper_cap_normals, normals)
    cap_mask = jp.logical_or(upper_cap_mask, lower_cap_mask)
    normals = jp.where(cap_mask, normals, shell_normals)
    return normals
