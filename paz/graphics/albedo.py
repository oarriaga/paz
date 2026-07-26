import jax
import jax.numpy as jp

import paz


def compute_shape_albedo(shape, material, points):
    pattern_colors = compute_pattern_colors(shape, points)
    material_colors = jp.full_like(points, material.color)
    return pattern_colors + material_colors


def compute_pattern_colors(shape, points):
    args = shape.pattern.transform, shape.transform, points
    pattern_points = compute_points_in_pattern(*args)
    cases = [
        paz.graphics.patterns.empty.compute_colors,
        paz.graphics.patterns.spherical.compute_colors,
        paz.graphics.patterns.planar.compute_colors,
        paz.graphics.patterns.cylindrical.compute_colors,
    ]
    switch_args = pattern_points, shape.pattern.image
    return jax.lax.switch(shape.pattern.type, cases, *switch_args)


def compute_points_in_pattern(pattern_transform, shape_transform, points):
    world_to_shape = jp.linalg.inv(shape_transform)
    points_shape = paz.algebra.transform_points(world_to_shape, points)
    shape_to_pattern = jp.linalg.inv(pattern_transform)
    return paz.algebra.transform_points(shape_to_pattern, points_shape)
