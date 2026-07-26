import jax
import jax.numpy as jp

import paz

from paz.graphics.geometry import (
    compute_hits_to_light,
    compute_reflections_dot_eye,
)


def compute_colors(albedo, material, points, normals, eye, light):
    ambient = compute_ambient(albedo, material, light)
    diffuse = compute_soft_diffuse(albedo, material, light, points, normals)
    specular = compute_specular(material, light, points, normals, eye)
    return ambient + diffuse + specular


def compute_colors_with_shadow(
    albedo, material, points, normals, eye, light, is_shadow
):
    colors = compute_colors(albedo, material, points, normals, eye, light)
    ambient = compute_ambient(albedo, material, light)
    is_shadow = jp.expand_dims(is_shadow, 1)
    return (ambient * is_shadow) + (colors * (1.0 - is_shadow))


def compute_ambient(albedo, material, light):
    return albedo * light.intensity * material.ambient


def compute_diffuse(albedo, material, light, points, normals):
    hits_to_light = compute_hits_to_light(light.position, points)
    lambertian = paz.algebra.dot(hits_to_light, normals)
    lambertian = jp.maximum(lambertian, 0.0)
    lambertian = jp.expand_dims(lambertian, -1)
    base_color = albedo * light.intensity
    return base_color * material.diffuse * lambertian


def compute_soft_diffuse(albedo, material, light, points, normals, slope=10.0):
    hits_to_light = compute_hits_to_light(light.position, points)
    dot_product = paz.algebra.dot(hits_to_light, normals)
    lambertian = jax.nn.softplus(dot_product * slope) / slope
    lambertian = jp.expand_dims(lambertian, -1)
    base_color = albedo * light.intensity
    return base_color * material.diffuse * lambertian


def compute_specular(material, light, points, normals, eye):
    reflections = compute_reflections_dot_eye(light, points, normals, eye)
    factor = jp.power(reflections, material.shininess)
    factor = jp.expand_dims(factor, -1)
    specular = light.intensity * material.specular * factor
    return specular
