import jax
import jax.numpy as jp
from functools import partial

from paz.graphics.constants import FARAWAY
import paz

from .geometry import build_edges


EPSILON = 1e-8


def compute_f(edges_AC, edges_AB, ray_directions):
    directions_cross_edges_AC = jp.cross(ray_directions, edges_AC)
    determinants = paz.algebra.dot(edges_AB, directions_cross_edges_AC)
    valid = jp.abs(determinants) > EPSILON
    safe_determinants = jp.where(valid, determinants, 1.0)
    f = jp.where(valid, 1.0 / safe_determinants, 0.0)
    return f, directions_cross_edges_AC


def intersect_canonical_mesh(vertices, faces, ray_origins, ray_directions):
    edges_AB, edges_AC, points_A = build_edges(vertices, faces)
    f, directions_cross_edges_AC = compute_f(edges_AC, edges_AB, ray_directions)
    points_1_to_origin = ray_origins - points_A
    u = f * paz.algebra.dot(points_1_to_origin, directions_cross_edges_AC)
    origins_cross_edge_1 = jp.cross(points_1_to_origin, edges_AB)
    v = f * paz.algebra.dot(ray_directions, origins_cross_edge_1)
    hit_mask_u = jp.logical_not(jp.logical_or(u < 0.0, u > 1.0))
    hit_mask_v = jp.logical_not(jp.logical_or(v < 0.0, (u + v) > 1.0))
    hit_mask = jp.logical_and(hit_mask_u, hit_mask_v)
    depth = f * paz.algebra.dot(edges_AC, origins_cross_edge_1)
    hit_mask = jp.logical_and(hit_mask, depth > EPSILON)
    depth = jp.where(hit_mask, depth, FARAWAY)
    return hit_mask, depth, u, v


def _pad_to_chunks(faces, chunk_size):
    remainder = faces.shape[0] % chunk_size
    if remainder == 0:
        return faces
    pad_size = chunk_size - remainder
    last = jp.repeat(faces[-1:], pad_size, axis=0)
    return jp.concatenate([faces, last], axis=0)


def _select_closest(hit_mask, depth, u, v):
    best = jp.argmin(depth, axis=0).astype(jp.int32)
    idx = jp.expand_dims(best, 0)
    take = lambda x: jp.take_along_axis(x, idx, 0)[0]
    return take(hit_mask), take(depth), take(u), take(v), best


def _update_best(carry, closest, offset):
    mask, depth, u, v, face_idx = carry
    c_mask, c_depth, c_u, c_v, c_local = closest
    closer = c_depth < depth
    global_idx = offset + c_local
    new_mask = jp.where(closer, c_mask, mask)
    new_depth = jp.where(closer, c_depth, depth)
    new_u = jp.where(closer, c_u, u)
    new_v = jp.where(closer, c_v, v)
    new_idx = jp.where(closer, global_idx, face_idx)
    return new_mask, new_depth, new_u, new_v, new_idx


@jax.checkpoint
def _chunk_intersect(vertices, chunk_faces, rays):
    args = (vertices, chunk_faces, *rays)
    return intersect_canonical_mesh(*args)


def _chunk_step(carry, xs, vertices, rays):
    chunk_faces, offset = xs
    hit_mask, depth, u, v = _chunk_intersect(vertices, chunk_faces, rays)
    closest = _select_closest(hit_mask, depth, u, v)
    return _update_best(carry, closest, offset), None


def _init_carry(num_rays):
    mask = jp.zeros(num_rays, dtype=bool)
    depth = jp.full(num_rays, FARAWAY)
    u = jp.zeros(num_rays)
    v = jp.zeros(num_rays)
    idx = jp.zeros(num_rays, dtype=jp.int32)
    return mask, depth, u, v, idx


def intersect_chunked(vertices, faces, rays, chunk_size=1024):
    num_faces = faces.shape[0]
    padded = _pad_to_chunks(faces, chunk_size)
    num_chunks = padded.shape[0] // chunk_size
    chunks = jp.reshape(padded, (num_chunks, chunk_size, 3))
    offsets = jp.arange(num_chunks, dtype=jp.int32) * chunk_size
    init = _init_carry(rays[0].shape[0])
    step = partial(_chunk_step, vertices=vertices, rays=rays)
    carry, _ = jax.lax.scan(step, init, (chunks, offsets))
    mask, depth, u, v, idx = carry
    idx = jp.minimum(idx, num_faces - 1)
    return mask, depth, u, v, idx


def intersect_mesh(mesh, ray_origins, ray_directions, chunk_size=1024):
    world_to_shape = jp.linalg.inv(mesh.transform)
    args = (world_to_shape, ray_origins, ray_directions)
    rays = paz.algebra.transform_rays(*args)
    return intersect_chunked(mesh.vertices, mesh.faces, rays, chunk_size)
