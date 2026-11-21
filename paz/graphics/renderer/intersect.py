from collections import namedtuple

import jax
import jax.numpy as jp

import paz

from paz.graphics.composite import find_closest_intersection_args, take_closest
from paz.graphics.constants import FARAWAY
from paz.graphics.renderer import csg
from paz.graphics.types import Surfaces

TRIANGLE_HIT_NAMES = "hit_mask depth points normals eyes albedo primitive "
TRIANGLE_HIT_NAMES += "face_index"
TriangleHit = namedtuple("TriangleHit", TRIANGLE_HIT_NAMES.split())


def build_candidates(compiled, rays, triangle_hit):
    rows = []
    if len(compiled.shapes) > 0:
        rows.append(intersect_shapes(compiled, rays))
    if triangle_hit is not None:
        rows.append(build_triangle_row(triangle_hit))
    joined = tuple(jp.concatenate(fields, axis=0) for fields in zip(*rows))
    hit_masks, depths, points, normals, eyes = joined
    return hit_masks, depths, Surfaces(points, normals, eyes)


def intersect_shapes(compiled, rays):
    shapes = paz.graphics.shapes
    merged = shapes.field_merge(compiled.shapes, ["transform", "type"])
    compute_roots = jax.vmap(paz.lock(shapes.compute_roots, *rays))
    hit_masks, roots, depths = compute_roots(merged)
    hit_masks, depths = replace_csg_hits(compiled, hit_masks, roots, depths)
    compute_surface = jax.vmap(paz.lock(shapes.compute_surface, *rays))
    surfaces = compute_surface(merged, roots, depths)
    hit_masks = jp.where(jp.expand_dims(compiled.mask, 1), hit_masks, False)
    return hit_masks, depths, *surfaces


def replace_csg_hits(compiled, hit_masks, roots, depths):
    for tree in compiled.csg_trees:
        leaves = jp.array(csg.collect_leaves(tree))
        csg_depths = csg.intersect(tree, roots[leaves], compiled.mask)
        hit_masks = hit_masks.at[leaves].set(csg_depths < FARAWAY)
        depths = depths.at[leaves].set(jp.expand_dims(csg_depths, -1))
    return hit_masks, depths


def build_triangle_row(triangle_hit):
    depth = jp.expand_dims(triangle_hit.depth, -1)
    fields = triangle_hit.hit_mask, depth, triangle_hit.points
    fields += triangle_hit.normals, triangle_hit.eyes
    return tuple(jp.expand_dims(field, 0) for field in fields)


def find_closest(hit_masks, depths, surfaces):
    indices = find_closest_intersection_args(hit_masks, depths)
    fields = hit_masks, depths, *surfaces
    closest = tuple(take_closest(field, indices) for field in fields)
    return paz.graphics.Hit(*closest, indices)


def compute_triangle_hit(compiled, rays, face_chunk):
    if compiled.triangles is None:
        triangle_hit = None
    else:
        triangle_hit = build_triangle_hit(compiled, rays, face_chunk)
    return triangle_hit


def build_triangle_hit(compiled, rays, face_chunk):
    triangles = compiled.triangles
    args = triangles, rays, face_chunk
    result = paz.graphics.mesh.intersect_triangles(*args)
    hit_mask, depth, points, normals, eyes, face_index, u, v = result
    primitive = triangles.primitive_index[face_index]
    hit_mask = jp.logical_and(hit_mask, compiled.triangle_mask[primitive])
    depth = jp.where(hit_mask, depth, paz.graphics.FARAWAY)
    albedo_args = triangles, face_index, u, v
    albedo = paz.graphics.albedo.compute_triangle_albedo(*albedo_args)
    args = hit_mask, depth, points, normals, eyes, albedo, primitive
    return TriangleHit(*args, face_index)


def intersect_shadow_groups(shapes, origins, directions):
    intersect_all = paz.graphics.shapes.intersect_all
    intersect_group = jax.vmap(paz.lock(intersect_all, origins, directions))
    rows = []
    for group, start_arg, final_arg in iterate_shape_groups(shapes):
        indices = jp.arange(start_arg, final_arg)
        rows.append((*intersect_group(group), indices))
    return tuple(jp.concatenate(fields, axis=0) for fields in zip(*rows))


def iterate_shape_groups(shapes):
    start_arg = 0
    for group in paz.graphics.shapes.group_by_pattern_size(shapes).values():
        final_arg = start_arg + len(group)
        yield paz.graphics.shapes.merge(*group), start_arg, final_arg
        start_arg = final_arg


def slice_surfaces(surfaces, start_arg, final_arg):
    points = surfaces.points[start_arg:final_arg]
    normals = surfaces.normals[start_arg:final_arg]
    eyes = surfaces.eyes[start_arg:final_arg]
    return Surfaces(points, normals, eyes)
