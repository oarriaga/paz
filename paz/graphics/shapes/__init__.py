from collections import defaultdict
import jax
import paz
import jax.numpy as jp
from paz.backend.algebra import transform_points
from paz.graphics.geometry import transform_rays, compute_points3D

from .sphere import intersect_canonical_sphere
from .cube import intersect_canonical_cube
from .cylinder import intersect_canonical_cylinder
from .cone import intersect_canonical_cone
from .plane import intersect_canonical_plane

from .sphere import compute_canonical_normals_sphere
from .cube import compute_canonical_normals_cube
from .cylinder import compute_canonical_normals_cylinder
from .cone import compute_canonical_normals_cone
from .plane import compute_canonical_normals_plane

from ..constants import EPSILON


intersection_cases = [
    intersect_canonical_sphere,
    intersect_canonical_cube,
    intersect_canonical_cylinder,
    intersect_canonical_cone,
    intersect_canonical_plane,
]

normal_cases = [
    compute_canonical_normals_sphere,
    compute_canonical_normals_cube,
    compute_canonical_normals_cylinder,
    compute_canonical_normals_cone,
    compute_canonical_normals_plane,
]


def _merge(*leafs):
    concatenated_leafs = []
    for leaf in leafs:
        concatenated_leafs.append(leaf)
    return jp.array(concatenated_leafs)


def merge(*shapes):
    return jax.tree.map(_merge, *shapes)


def _expand_leafs(leaf):
    return jp.expand_dims(jp.array(leaf), 0)


def expand(shape):
    return jax.tree.map(_expand_leafs, shape)


def get_num_shapes(shapes):
    """Computes the number of shapes from a merged or expanded Shape PyTree."""
    return shapes.transform.shape[0]


def _append(leaf_A, leaf_B):
    leaf_B = jp.expand_dims(leaf_B, axis=0)
    return jp.concatenate([leaf_A, leaf_B], axis=0)


def append(shapes, shape):
    return jax.tree.map(_append, shapes, shape)


def _concatenate(*leaves):
    return jp.concatenate(list(leaves), axis=0)


def concatenate(*batched_shapes):
    return jax.tree.map(_concatenate, *batched_shapes)


def split(shape):
    """Splits a batched Shape object into a list of N shapes with batch 1."""
    num = shape.transform.shape[0]
    return [jax.tree.map(lambda x: x[i : (i + 1)], shape) for i in range(num)]


def intersect(shape, ray_origins, ray_directions):
    world_to_shape = jp.linalg.inv(shape.transform)
    # world_to_shape = paz.SE3.invert(shape.transform)
    rays_shape = transform_rays(world_to_shape, ray_origins, ray_directions)
    intersections = jax.lax.switch(shape.type, intersection_cases, *rays_shape)
    hit_mask, sorted_depths, depth = intersections
    # transform world points
    world_points = compute_points3D(ray_origins, ray_directions, depth)
    # world_to_shape = jp.linalg.inv(shape.transform)
    # world_to_shape = paz.SE3.invert(shape.transform)
    shape_points = transform_points(world_to_shape, world_points)
    # transform world normals
    shape_normals = jax.lax.switch(shape.type, normal_cases, shape_points)
    world_normals = transform_points(world_to_shape.T, shape_normals)
    world_normals = paz.algebra.normalize(world_normals)
    # postprocess normals
    eyes = compute_eyes(ray_directions)
    world_normals = invert_inside_normals(eyes, world_normals)
    world_points = move_toward_normals(world_points, world_normals)
    return hit_mask, depth, world_points, world_normals, eyes


def intersect_all(shape, ray_origins, ray_directions):
    world_to_shape = jp.linalg.inv(shape.transform)
    rays_shape = transform_rays(world_to_shape, ray_origins, ray_directions)
    intersections = jax.lax.switch(shape.type, intersection_cases, *rays_shape)
    hit_mask, depths, depth = intersections
    world_points = compute_points3D(ray_origins, ray_directions, depth)
    shape_points = transform_points(world_to_shape, world_points)
    shape_normals = jax.lax.switch(shape.type, normal_cases, shape_points)
    world_normals = transform_points(world_to_shape.T, shape_normals)
    world_normals = paz.algebra.normalize(world_normals)
    eyes = compute_eyes(ray_directions)
    world_normals = invert_inside_normals(eyes, world_normals)
    world_points = move_toward_normals(world_points, world_normals)
    return hit_mask, depths, world_points, world_normals, eyes


def compute_eyes(ray_directions):
    eyes = -ray_directions
    return eyes


def invert_inside_normals(eyes, normals):
    inside_mask = paz.algebra.dot(normals, eyes) < 0
    inside_mask = jp.expand_dims(inside_mask, 1)
    normals = jp.where(inside_mask, -normals, normals)
    return normals


def move_toward_normals(points, normals):
    return points + (normals * EPSILON)  # fix pattern noise


def group_by_pattern_size(shapes_list):
    """Groups a list of Shape objects by the size of their pattern's image."""
    grouped_shapes = defaultdict(list)
    for shape in shapes_list:
        image_size_key = shape.pattern.image.shape[:2]
        grouped_shapes[image_size_key].append(shape)
    return dict(grouped_shapes)


def pad_depths(depths, hit_mask, num_pad_rows):
    hit_mask = jp.expand_dims(hit_mask, axis=0)  # (2, N)
    depths = jp.where(hit_mask, depths, paz.graphics.FARAWAY)
    if num_pad_rows != 0:
        mask_shape = (num_pad_rows, depths.shape[1])
        mask = jp.full(mask_shape, paz.graphics.FARAWAY)
        depths = jp.vstack([depths, mask])
    return depths
