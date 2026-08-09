import jax.numpy as jp

import paz

BOUNCE_ORIGIN_EPSILON = 1e-2


def compute_bounce(directions, closest, refractive_index, material):
    args = directions, closest.normal, refractive_index
    terms = compute_refraction_terms(*args, material.refractive_indices)
    normal, eye, n_1, n_2, n_ratio = terms
    reflectance = compute_reflectance(normal, eye, n_1, n_2)
    ray_args = normal, eye, n_ratio, closest.point, material.transparencies
    return compute_new_rays(*ray_args), n_2, reflectance


def compute_refraction_terms(directions, normal, n_1, refractive_indices):
    eye = -directions
    normal, is_inside = flip_normal_if_inside(eye, normal)
    # TODO why 1.0 hardcoded
    n_2 = jp.where(is_inside, 1.0, refractive_indices)
    return normal, eye, n_1, n_2, n_1 / n_2


def flip_normal_if_inside(eye, normal):
    is_inside = jp.sum(normal * eye, axis=-1) < 0.0
    return jp.where(jp.expand_dims(is_inside, -1), -normal, normal), is_inside


def compute_reflectance(normal, eye, n_1, n_2):
    cosines = compute_transmission_cosines(eye, normal, n_1 / n_2)
    cos_incident, sin_transmit_squared, cos_transmit = cosines
    cos = jp.where(n_1 > n_2, cos_transmit, cos_incident)
    base_reflectance = ((n_1 - n_2) / (n_1 + n_2)) ** 2
    grazing = (1.0 - cos) ** 5
    reflectance = base_reflectance + (1.0 - base_reflectance) * grazing
    return jp.where(sin_transmit_squared > 1.0, 1.0, reflectance)


def compute_transmission_cosines(eye, normal, n_ratio):
    cos_incident = jp.sum(eye * normal, axis=-1)
    sin_transmit_squared = (n_ratio**2) * (1.0 - (cos_incident**2))
    cos_transmit = jp.sqrt(jp.maximum(0.0, 1.0 - sin_transmit_squared))
    return cos_incident, sin_transmit_squared, cos_transmit


def compute_new_rays(normal, eye, n_ratio, point, transparencies):
    is_transparent = transparencies > 0.0
    do_reflect = jp.expand_dims(~is_transparent, -1)
    reflection = paz.graphics.geometry.reflect(-eye, normal)
    refraction = compute_refractive_direction(eye, normal, n_ratio)
    direction = jp.where(do_reflect, reflection, refraction)
    lower_point, upper_point = displace_by_normal(point, normal)
    origin = jp.where(do_reflect, upper_point, lower_point)
    return origin, paz.algebra.normalize(direction)


def compute_refractive_direction(eye, normal, n_ratio):
    args = eye, normal, n_ratio
    cos_incident, _, cos_transmit = compute_transmission_cosines(*args)
    inside_vector = -eye * jp.expand_dims(n_ratio, -1)
    up_weight = n_ratio * cos_incident - cos_transmit
    return jp.expand_dims(up_weight, -1) * normal + inside_vector


def displace_by_normal(point, normal):
    upper_point = point + normal * BOUNCE_ORIGIN_EPSILON
    lower_point = point - normal * BOUNCE_ORIGIN_EPSILON
    return lower_point, upper_point


def compute_bounce_factor(material, reflectance):
    is_transparent = material.transparencies > 0.0
    is_reflective = material.reflectivities > 0.0
    transparent_factor = material.transparencies * (1.0 - reflectance)
    reflective_factor = jp.where(is_reflective, material.reflectivities, 0.0)
    return jp.where(is_transparent, transparent_factor, reflective_factor)
