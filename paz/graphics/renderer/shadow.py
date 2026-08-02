import jax
import jax.numpy as jp

import paz

from paz.graphics.composite import compute_scene_hit_mask
from paz.graphics.renderer.intersect import intersect_shadow_groups

SHADOW_ORIGIN_EPSILON = 1e-5
SHADOW_SELF_HIT_EPSILON = 1e-5


def compute_occlusion(compiled, closest, indices, directions, distance):
    origins = compute_shadow_ray_origins(closest.point, closest.normal)
    shadow_args = compiled.shapes, origins, directions
    intersections = intersect_shadow_groups(*shadow_args)
    hit_masks, depths, _, _, _, casters = intersections
    masks = resolve_shadow_masks(compiled, hit_masks)
    depth_args = masks, depths, casters, indices, closest.normal, directions
    masks, depths = select_shadow_depths(*depth_args)
    return compute_soft_occlusion(masks, depths, distance)


def compute_shadow_ray_origins(points, normals):
    over_point, _ = compute_surface_points(points, normals)
    return over_point


def compute_surface_points(point, normal):
    over_point = point + normal * SHADOW_ORIGIN_EPSILON
    under_point = point - normal * SHADOW_ORIGIN_EPSILON
    return over_point, under_point


def resolve_shadow_masks(compiled, hit_masks):
    transparencies = compute_transparencies(compiled.shapes)
    masks = jp.where(jp.expand_dims(compiled.mask, 1), hit_masks, False)
    masks = hide_transparent_shapes(masks, transparencies > 0.0)
    if compiled.shadow_mask is None:
        resolved = masks
    else:
        resolved = hide_non_casting_shapes(masks, compiled.shadow_mask)
    return resolved


def compute_transparencies(shapes):
    return jp.array([shape.material.transparency for shape in shapes])


def hide_transparent_shapes(shadow_masks, is_transparent):
    return jp.where(jp.expand_dims(is_transparent, 1), False, shadow_masks)


def hide_non_casting_shapes(shadow_masks, shadow_mask):
    return jp.where(jp.expand_dims(shadow_mask, 1), shadow_masks, False)


def select_shadow_depths(
    hit_masks, depths, casters, receivers, normals, directions
):
    same_shape = casters[:, None] == receivers[None, :]
    front_args = same_shape, normals, directions
    front_side_hits = compute_front_side_shadow_mask(*front_args)
    root_args = hit_masks, depths, same_shape, front_side_hits
    valid_roots = compute_valid_roots(*root_args)
    depths = jp.where(valid_roots, depths, paz.graphics.FARAWAY)
    return jp.any(valid_roots, axis=1), jp.min(depths, axis=1)


def compute_front_side_shadow_mask(same_shape, normals, directions):
    front_side = paz.algebra.dot(normals, directions) >= 0.0
    return jp.logical_and(same_shape, front_side[None, :])


def compute_valid_roots(hit_masks, depths, same_shape, front_side_hits):
    thresholds = compute_shadow_depth_thresholds(same_shape)
    valid_roots = depths > thresholds[:, None, :]
    valid_roots = jp.logical_and(valid_roots, depths < paz.graphics.FARAWAY)
    valid_roots = jp.logical_and(jp.expand_dims(hit_masks, 1), valid_roots)
    return jp.logical_and(valid_roots, ~front_side_hits[:, None, :])


def compute_shadow_depth_thresholds(same_shape):
    return jp.where(same_shape, SHADOW_SELF_HIT_EPSILON, paz.graphics.EPSILON)


def compute_soft_occlusion(hit_masks, depths, light_lengths, slope=0.01):
    closest_depths = jp.where(hit_masks, depths, paz.graphics.FARAWAY)
    closest_depths = jp.min(closest_depths, axis=0)
    scene_hit_mask = compute_scene_hit_mask(hit_masks)
    blockers = closest_depths <= light_lengths
    blocker_mask = jp.logical_and(scene_hit_mask, blockers)
    difference = light_lengths - closest_depths
    occlusion = jax.nn.sigmoid(slope * difference)
    return jp.where(blocker_mask, occlusion, 0.0)
