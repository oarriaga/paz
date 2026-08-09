from collections import namedtuple

import jax.numpy as jp

import paz

from paz.graphics.composite import compute_scene_hit_mask
from paz.graphics.renderer.intersect import intersect_shadow_groups

SHADOW_ORIGIN_EPSILON = 1e-5
SHADOW_SELF_HIT_EPSILON = 1e-5
# A shadow ray leaving a mesh surface can re-hit its own triangle at a
# grazing angle. Shapes reject that by identity; triangles have no such
# index, so they need a distance that clears float error at scene scale.
TRIANGLE_SELF_HIT_EPSILON = 1e-3
NO_SHAPE = -1

Receiver = namedtuple("Receiver", ["points", "normals", "indices"])


def build_shape_receiver(closest, indices):
    return Receiver(closest.point, closest.normal, indices)


def build_triangle_receiver(triangle_hit):
    indices = jp.full(len(triangle_hit.points), NO_SHAPE)
    return Receiver(triangle_hit.points, triangle_hit.normals, indices)


def compute_occlusion(compiled, receiver, light, face_chunk):
    directions, distance = compute_light_directions(light, receiver.points)
    origins = compute_shadow_ray_origins(receiver.points, receiver.normals)
    rows = []
    if len(compiled.shapes) > 0:
        shape_args = compiled, receiver, origins, directions
        rows.append(compute_shape_blockers(*shape_args))
    if compiled.triangles is not None:
        blocker_args = compiled, origins, directions, face_chunk
        rows.append(compute_triangle_blockers(*blocker_args))
    masks = jp.concatenate([row[0] for row in rows], axis=0)
    depths = jp.concatenate([row[1] for row in rows], axis=0)
    return compute_occlusion_mask(masks, depths, distance)


def compute_light_directions(light, points):
    vector = light.position - points
    norm = paz.algebra.compute_norms(vector, 1)
    return vector / norm, jp.squeeze(norm, axis=1)


def compute_shape_blockers(compiled, receiver, origins, directions):
    shadow_args = compiled.shapes, origins, directions
    intersections = intersect_shadow_groups(*shadow_args)
    hit_masks, depths, _, _, _, casters = intersections
    masks = resolve_shadow_masks(compiled, hit_masks)
    depth_args = masks, depths, casters, receiver.indices
    depth_args += receiver.normals, directions
    return select_shadow_depths(*depth_args)


def compute_triangle_blockers(compiled, origins, directions, face_chunk):
    triangles = compiled.triangles
    args = triangles.vertices, triangles.faces, (origins, directions)
    result = paz.graphics.mesh.intersect_chunked(*args, face_chunk)
    hit_mask, depth, _, _, face_index = result
    primitive = triangles.primitive_index[face_index]
    hit_mask = jp.logical_and(hit_mask, compiled.triangle_mask[primitive])
    hit_mask = jp.logical_and(hit_mask, depth > TRIANGLE_SELF_HIT_EPSILON)
    hit_mask = hide_non_casting_triangles(compiled, hit_mask, primitive)
    depth = jp.where(hit_mask, depth, paz.graphics.FARAWAY)
    return jp.expand_dims(hit_mask, 0), jp.expand_dims(depth, 0)


def hide_non_casting_triangles(compiled, hit_mask, primitive):
    if compiled.triangle_shadow_mask is None:
        casting = hit_mask
    else:
        casting = compiled.triangle_shadow_mask[primitive]
        casting = jp.logical_and(hit_mask, casting)
    return casting


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


def compute_occlusion_mask(hit_masks, depths, light_lengths):
    closest_depths = jp.where(hit_masks, depths, paz.graphics.FARAWAY)
    closest_depths = jp.min(closest_depths, axis=0)
    scene_hit_mask = compute_scene_hit_mask(hit_masks)
    blockers = closest_depths <= light_lengths
    blocker_mask = jp.logical_and(scene_hit_mask, blockers)
    return jp.where(blocker_mask, 1.0, 0.0)
