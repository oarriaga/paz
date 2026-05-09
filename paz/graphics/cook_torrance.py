from collections import namedtuple

import jax.numpy as jp

import paz

from paz.graphics.geometry import compute_hits_to_light
from paz.graphics.phong import compute_base_color


Cosines = namedtuple(
    "Cosines",
    [
        "normal_dot_light",
        "normal_dot_view",
        "normal_dot_half",
        "view_dot_half",
    ],
)


def compute_halfway_direction(view, light_direction):
    return paz.algebra.normalize(view + light_direction)


def compute_directional_cosines(eye, normals, light, points):
    light_direction = compute_hits_to_light(light.position, points)
    view = paz.algebra.normalize(eye)
    halfway = compute_halfway_direction(view, light_direction)
    cosines = (
        jp.maximum(paz.algebra.dot(normals, light_direction), 0.0),
        jp.maximum(paz.algebra.dot(normals, view), 1e-4),
        jp.maximum(paz.algebra.dot(normals, halfway), 0.0),
        jp.maximum(paz.algebra.dot(view, halfway), 0.0),
    )
    return Cosines(*cosines)


def compute_microfacet_distribution(normal_dot_half, roughness):
    roughness_squared = roughness * roughness
    alpha = roughness_squared * roughness_squared
    base = normal_dot_half * normal_dot_half * (alpha - 1.0) + 1.0
    return alpha / (jp.pi * base * base + 1e-7)


def compute_visibility(cosine, roughness):
    factor = (roughness + 1.0) ** 2 / 8.0
    return cosine / (cosine * (1.0 - factor) + factor + 1e-7)


def compute_geometry_term(normal_dot_light, normal_dot_view, roughness):
    light_visibility = compute_visibility(normal_dot_light, roughness)
    view_visibility = compute_visibility(normal_dot_view, roughness)
    return light_visibility * view_visibility


def compute_fresnel_reflectance(view_dot_half, base_reflectance):
    grazing = jp.power(1.0 - view_dot_half, 5.0)
    return base_reflectance + (1.0 - base_reflectance) * grazing


def compute_base_reflectance(material):
    dielectric = jp.full_like(material.color, material.base_reflectance)
    metallic = material.metallic
    return dielectric * (1.0 - metallic) + material.color * metallic


def compute_specular_color(material, reflectance, cosines):
    distribution = compute_microfacet_distribution(
        cosines.normal_dot_half, material.roughness
    )
    geometry = compute_geometry_term(
        cosines.normal_dot_light, cosines.normal_dot_view, material.roughness
    )
    denominator = 4.0 * cosines.normal_dot_light * cosines.normal_dot_view
    factor = distribution * geometry / (denominator + 1e-7)
    return reflectance * jp.expand_dims(factor, -1)


def compute_diffuse_color(material, base_color, reflectance):
    base = base_color / jp.pi
    energy = (1.0 - reflectance) * (1.0 - material.metallic)
    return base * energy


def compute_direct_lighting(material, base_color, points, normals, eye, light):
    cosines = compute_directional_cosines(eye, normals, light, points)
    view_dot_half = jp.expand_dims(cosines.view_dot_half, -1)
    base_reflectance = compute_base_reflectance(material)
    reflectance = compute_fresnel_reflectance(view_dot_half, base_reflectance)
    diffuse = compute_diffuse_color(material, base_color, reflectance)
    specular = compute_specular_color(material, reflectance, cosines)
    visible = jp.expand_dims(cosines.normal_dot_light, -1)
    return (diffuse + specular) * visible * light.intensity


def compute_ambient(shape, material, light, points):
    base_color = compute_base_color(shape, material, light, points)
    return base_color * material.ambient


def compute_colors(shape, material, points, normals, eye, light):
    base_color = compute_base_color(shape, material, light, points)
    ambient = base_color * material.ambient
    direct = compute_direct_lighting(
        material, base_color, points, normals, eye, light
    )
    return ambient + direct


def compute_colors_with_shadow(
    shape, material, points, normals, eye, light, is_shadow
):
    full = compute_colors(shape, material, points, normals, eye, light)
    ambient = compute_ambient(shape, material, light, points)
    is_shadow = jp.expand_dims(is_shadow, 1)
    return ambient * is_shadow + full * (1.0 - is_shadow)
