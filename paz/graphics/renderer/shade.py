import jax
import jax.numpy as jp

import paz

from paz.graphics.composite import take_closest
from paz.graphics.renderer import shadow
from paz.graphics.renderer.intersect import (
    iterate_shape_groups,
    slice_surfaces,
)


def compute_hit_colors(
    compiled, closest, surfaces, shadows, triangle_hit, face_chunk
):
    if len(compiled.shapes) == 0:
        colors = compute_triangle_colors(compiled, triangle_hit)
    elif triangle_hit is None:
        shape_args = compiled, closest, surfaces, shadows, face_chunk
        colors = compute_shape_colors(*shape_args)
    else:
        args = compiled, closest, surfaces, shadows, triangle_hit
        colors = blend_hit_colors(*args, face_chunk)
    return colors


def blend_hit_colors(
    compiled, closest, surfaces, shadows, triangle_hit, face_chunk
):
    # TODO a mixed scene shades both paths for every ray and throws one
    # away. Joining them before selection needs color_with_shadows to
    # return rows instead of selecting inside its per-light scan.
    shape_args = compiled, closest, surfaces, shadows, face_chunk
    shape_colors = compute_shape_colors(*shape_args)
    triangle_colors = compute_triangle_colors(compiled, triangle_hit)
    is_triangle = closest.primitive_index == len(compiled.shapes)
    is_triangle = jp.expand_dims(is_triangle, -1)
    return jp.where(is_triangle, triangle_colors, shape_colors)


def compute_shape_colors(compiled, closest, surfaces, shadows, face_chunk):
    num_shapes = len(compiled.shapes)
    indices = jp.minimum(closest.primitive_index, num_shapes - 1)
    surfaces = slice_surfaces(surfaces, 0, num_shapes)
    if shadows:
        args = compiled, closest, surfaces, indices, face_chunk
        colors = color_with_shadows(*args)
    else:
        colors = color_without_shadow(compiled, surfaces, indices)
    return colors


def compute_triangle_colors(compiled, triangle_hit):
    materials = compiled.triangles.materials
    material = gather_triangle_material(materials, triangle_hit.primitive)
    shader = select_shader(materials)
    colors = jp.zeros_like(triangle_hit.albedo)
    for light in compiled.lights:
        args = triangle_hit.albedo, material, triangle_hit.points
        args += triangle_hit.normals, triangle_hit.eyes, light
        colors = colors + shader.compute_colors(*args)
    return colors


def gather_triangle_material(materials, primitive):
    material = jax.tree.map(lambda field: field[primitive], materials)
    return jax.tree.map(expand_scalar_field, material)


def expand_scalar_field(field):
    if field.ndim == 1:
        field = jp.expand_dims(field, -1)
    return field


def select_shader(material):
    if isinstance(material, paz.graphics.CookTorranceMaterial):
        shader = paz.graphics.cook_torrance
    else:
        shader = paz.graphics.phong
    return shader


def color_without_shadow(compiled, surfaces, indices):
    colors = []
    lights = paz.graphics.shapes.merge(*compiled.lights)
    for group, start_arg, final_arg in iterate_shape_groups(compiled.shapes):
        group_surfaces = slice_surfaces(surfaces, start_arg, final_arg)
        albedo = compute_group_albedo(group, group_surfaces.points)
        shader = select_shader(group.material)
        axes = 0, 0, 0, 0, 0, None
        color_per_light = jax.vmap(shader.compute_colors, axes)
        color = jax.vmap(color_per_light, (None, None, None, None, None, 0))
        args = albedo, group.material, *group_surfaces, lights
        colors.append(jp.sum(color(*args), axis=0))
    return take_closest(jp.concatenate(colors, axis=0), indices)


def compute_group_albedo(group, points):
    compute_albedo = paz.graphics.albedo.compute_shape_albedo
    return jax.vmap(compute_albedo)(group, group.material, points)


def color_with_shadows(compiled, closest, surfaces, indices, face_chunk):
    colors = jp.zeros((len(surfaces.points[0]), 3))
    lights = paz.graphics.shapes.merge(*compiled.lights)
    step_args = compiled, closest, surfaces, indices, face_chunk
    body = paz.lock(scan_light_step, *step_args)
    return jax.lax.scan(body, colors, lights)[0]


def scan_light_step(
    colors, light, compiled, closest, surfaces, indices, face_chunk
):
    args = compiled, closest, surfaces, indices, light, face_chunk
    return colors + compute_light_colors(*args), None


def compute_light_colors(
    compiled, closest, surfaces, indices, light, face_chunk
):
    directions, distance = compute_light_directions(light, closest.point)
    occlusion_args = compiled, closest, indices, directions, distance
    is_shadow = shadow.compute_occlusion(*occlusion_args, face_chunk)
    color_args = compiled.shapes, light, surfaces, is_shadow
    return take_closest(compute_shadowed_colors(*color_args), indices)


def compute_light_directions(light, points):
    vector = light.position - points
    norm = paz.algebra.compute_norms(vector, 1)
    return vector / norm, jp.squeeze(norm, axis=1)


def compute_shadowed_colors(shapes, light, surfaces, is_shadow):
    colors = []
    for group, start_arg, final_arg in iterate_shape_groups(shapes):
        group_surfaces = slice_surfaces(surfaces, start_arg, final_arg)
        albedo = compute_group_albedo(group, group_surfaces.points)
        shader = select_shader(group.material)
        axes = 0, 0, 0, 0, 0, None, None
        color = jax.vmap(shader.compute_colors_with_shadow, axes)
        args = albedo, group.material, *group_surfaces, light, is_shadow
        colors.append(color(*args))
    return jp.concatenate(colors, axis=0)
