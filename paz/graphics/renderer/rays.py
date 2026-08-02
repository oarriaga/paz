from collections import namedtuple

import jax
import jax.numpy as jp

import paz

from paz.graphics.renderer import intersect, material, optics, shade

STATE_NAMES = "color depth hit_mask throughput active_mask "
STATE_NAMES += "refractive_index rays"
RenderState = namedtuple("RenderState", STATE_NAMES.split())


def render(rays, compiled, shadows, num_bounces, face_chunk):
    state = initialize_state(rays)
    step = paz.lock(bounce, compiled, shadows, face_chunk)
    for step_arg in range(num_bounces):
        state = step(state, step_arg)
    return state.hit_mask, state.depth, state.color


def bounce(state, step_arg, compiled, shadows, face_chunk):
    hit_args = compiled, state.rays, face_chunk
    triangle_hit = intersect.compute_triangle_hit(*hit_args)
    candidates = intersect.build_candidates(compiled, state.rays, triangle_hit)
    hit_masks, depths, surfaces = candidates
    closest = intersect.find_closest(hit_masks, depths, surfaces)
    state = update_first_hit(state, closest, step_arg)
    state = update_active_mask(state, closest)
    color_args = compiled, closest, surfaces, shadows, triangle_hit
    colors = shade.compute_hit_colors(*color_args)
    return advance(state, compiled, closest, colors, triangle_hit)


def advance(state, compiled, closest, hit_colors, triangle_hit):
    material_args = compiled, closest, triangle_hit
    properties = material.compute_material_properties(*material_args)
    color_args = state.color, state.throughput, state.active_mask, hit_colors
    color_args += properties.reflectivities, properties.transparencies
    color = accumulate_color(*color_args)
    bounce_args = state.rays[1], closest, state.refractive_index, properties
    new_rays, n_2, reflectance = optics.compute_bounce(*bounce_args)
    args = new_rays, n_2, properties, reflectance
    return apply_bounce_update(state._replace(color=color), *args)


def render_chunks(rays, compiled, shadows, num_bounces, face_chunk, chunk):
    ray_chunks = split_ray_chunks(rays, chunk)
    step_args = compiled, shadows, num_bounces, face_chunk
    chunk_step = paz.lock(render_chunk_step, *step_args)
    hit_mask, depth, color = jax.lax.scan(chunk_step, None, ray_chunks)[1]
    return flatten_chunk_results(hit_mask, depth, color, len(rays[0]))


def render_chunk_step(carry, rays, compiled, shadows, num_bounces, face_chunk):
    args = rays, compiled, shadows, num_bounces, face_chunk
    return carry, render(*args)


def initialize_state(rays):
    num_rays = rays[0].shape[0]
    color = jp.zeros((num_rays, 3))
    depth = jp.full((num_rays,), paz.graphics.FARAWAY)
    hit_mask = jp.zeros((num_rays,), dtype=bool)
    throughput = jp.ones((num_rays, 3))
    active_mask = jp.ones((num_rays,), dtype=bool)
    refractive_index = jp.ones((num_rays,))
    fields = color, depth, hit_mask, throughput, active_mask
    return RenderState(*fields, refractive_index, rays)


def update_first_hit(state, closest, step_arg):
    if step_arg == 0:
        depth, hit_mask = closest.depth, closest.hit_mask
    else:
        depth, hit_mask = state.depth, state.hit_mask
    return state._replace(depth=depth, hit_mask=hit_mask)


def update_active_mask(state, closest):
    active_mask = state.active_mask & closest.hit_mask
    return state._replace(active_mask=active_mask)


def accumulate_color(
    colors, throughput, active_mask, hit_colors, reflectivities, transparencies
):
    weights = jp.maximum(1.0 - reflectivities - transparencies, 0.0)
    weights = jp.expand_dims(weights, -1)
    active_mask = jp.expand_dims(active_mask, -1)
    return colors + (throughput * active_mask * weights * hit_colors)


def apply_bounce_update(state, new_rays, n_2, properties, reflectance):
    is_transparent = properties.transparencies > 0.0
    is_reflective = properties.reflectivities > 0.0
    factor = optics.compute_bounce_factor(properties, reflectance)
    factor = jp.where(is_transparent & (reflectance >= 1.0), 1.0, factor)
    throughput = state.throughput * jp.expand_dims(factor, -1)
    active_mask = state.active_mask & (is_transparent | is_reflective)
    index_args = state, n_2, is_transparent, reflectance
    refractive_index = update_refractive_index(*index_args)
    args = throughput, active_mask, refractive_index, new_rays
    return replace_bounce_state(state, *args)


def update_refractive_index(state, n_2, is_transparent, reflectance):
    update_mask = is_transparent & (reflectance < 1.0)
    return jp.where(update_mask, n_2, state.refractive_index)


def replace_bounce_state(state, throughput, active_mask, index, new_rays):
    state = state._replace(throughput=throughput, active_mask=active_mask)
    return state._replace(refractive_index=index, rays=new_rays)


def split_ray_chunks(rays, chunk_size):
    origins, directions = rays
    origins = pad_to_chunks(origins, chunk_size)
    directions = pad_to_chunks(directions, chunk_size)
    num_chunks = origins.shape[0] // chunk_size
    shape = num_chunks, chunk_size, 3
    return origins.reshape(shape), directions.reshape(shape)


def pad_to_chunks(array, chunk_size):
    remainder = array.shape[0] % chunk_size
    if remainder == 0:
        padded = array
    else:
        padding = jp.repeat(array[-1:], chunk_size - remainder, axis=0)
        padded = jp.concatenate([array, padding], axis=0)
    return padded


def flatten_chunk_results(hit_mask, depth, color, num_rays):
    hit_mask = flatten_chunk_array(hit_mask, num_rays)
    depth = flatten_chunk_array(depth, num_rays)
    color = flatten_chunk_array(color, num_rays)
    return hit_mask, depth, color


def flatten_chunk_array(array, num_rays):
    shape = (-1,) + array.shape[2:]
    return array.reshape(shape)[:num_rays]
