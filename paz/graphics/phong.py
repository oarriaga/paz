import jax
import jax.numpy as jp

import paz

from paz.graphics.geometry import (
    compute_hits_to_light,
    compute_reflections_dot_eye,
)


def compute_colors_in_shape(pattern_transform, shape_transform, points_world):
    # TODO rename function
    world_to_shape = jp.linalg.inv(shape_transform)
    points_shape = paz.algebra.transform_points(world_to_shape, points_world)
    shape_to_pattern = jp.linalg.inv(pattern_transform)
    points_pattern = paz.algebra.transform_points(
        shape_to_pattern, points_shape
    )
    return points_pattern


def compute_pattern_colors(shape, points):
    pattern_transform = shape.pattern.transform
    pattern_image = shape.pattern.image
    points = compute_colors_in_shape(pattern_transform, shape.transform, points)
    cases = [
        paz.graphics.patterns.empty.compute_colors,
        paz.graphics.patterns.spherical.compute_colors,
        paz.graphics.patterns.planar.compute_colors,
        paz.graphics.patterns.cylindrical.compute_colors,
    ]
    pattern_colors = jax.lax.switch(
        shape.pattern.type, cases, points, pattern_image
    )
    return pattern_colors


def compute_material_colors(material, points):
    return jp.full_like(points, material.color)


def compute_base_color(shape, material, light, points):
    pattern_colors = compute_pattern_colors(shape, points)
    material_colors = compute_material_colors(material, points)
    # Align shape shading with mesh shading: light scales the combined base.
    base_colors = pattern_colors + material_colors
    return base_colors * light.intensity


def compute_ambient(shape, material, light, points):
    base_color = compute_base_color(shape, material, light, points)
    ambient = base_color * material.ambient
    return ambient


def compute_diffuse(shape, material, light, points, normals):
    hits_to_light = compute_hits_to_light(light.position, points)
    lambertian = paz.algebra.dot(hits_to_light, normals)
    lambertian = jp.maximum(lambertian, 0.0)
    lambertian = jp.expand_dims(lambertian, -1)
    base_color = compute_base_color(shape, material, light, points)
    return base_color * material.diffuse * lambertian


def compute_soft_diffuse(shape, material, light, points, normals, slope=10.0):
    hits_to_light = compute_hits_to_light(light.position, points)
    dot_product = paz.algebra.dot(hits_to_light, normals)
    lambertian = jax.nn.softplus(dot_product * slope) / slope
    lambertian = jp.expand_dims(lambertian, -1)
    base_color = compute_base_color(shape, material, light, points)
    return base_color * material.diffuse * lambertian


def compute_specular(material, light, points, normals, eye):
    reflections = compute_reflections_dot_eye(light, points, normals, eye)
    factor = jp.power(reflections, material.shininess)
    factor = jp.expand_dims(factor, -1)
    specular = light.intensity * material.specular * factor
    return specular


def compute_colors(shape, material, points, normals, eye, light):
    ambient = compute_ambient(shape, material, light, points)
    diffuse = compute_soft_diffuse(shape, material, light, points, normals)
    specular = compute_specular(material, light, points, normals, eye)
    return ambient + diffuse + specular


def compute_colors_with_shadow(
    shape, material, points, normals, eye, light, is_shadow
):
    ambient = compute_ambient(shape, material, light, points)
    diffuse = compute_soft_diffuse(shape, material, light, points, normals)
    specular = compute_specular(material, light, points, normals, eye)
    colors = ambient + diffuse + specular
    is_shadow = jp.expand_dims(is_shadow, 1)
    # color = jp.where(is_shadow, ambient, colors)
    # Use the occlusion_factor to smoothly blend between the full color and ambient
    color = (ambient * is_shadow) + (colors * (1.0 - is_shadow))
    return color
