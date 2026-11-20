import jax.numpy as jp

from paz.graphics.geometry import apply_hit_mask, is_ahead
from paz.graphics.constants import FARAWAY


def intersect_canonical_plane(ray_origins, ray_directions):
    ray_origins_y = ray_origins[:, 1]
    ray_directions_y = ray_directions[:, 1]
    sign = jp.where(ray_directions_y >= 0.0, 1e-7, -1e-7)
    safe_y = jp.where(jp.abs(ray_directions_y) > 1e-7, ray_directions_y, sign)
    roots = -ray_origins_y / safe_y
    # TODO check why jp.abs is not providing expected results for hit mask
    hit_mask = is_ahead(roots)
    depths = jp.expand_dims(apply_hit_mask(hit_mask, roots), -1)
    # The plane bounds the half-space below it, which a ray heading down
    # never leaves: its second root is at infinity.
    final_roots = jp.where(safe_y < 0.0, jp.inf, FARAWAY)
    no_roots = jp.full_like(roots, FARAWAY)
    all_roots = jp.vstack([roots, final_roots, no_roots, no_roots])
    return hit_mask, all_roots, depths


def compute_canonical_normals_plane(shape_points):
    shape_normals = jp.array([[0.0, 1.0, 0.0]])
    shape_normals = jp.repeat(shape_normals, len(shape_points), axis=0)
    return shape_normals
