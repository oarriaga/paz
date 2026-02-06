import jax.numpy as jp

from paz.graphics.constants import FARAWAY
import paz

from .geometry import build_edges


EPSILON = 1e-8


def compute_f(edges_AC, edges_AB, ray_directions):
    directions_cross_edges_AC = jp.cross(ray_directions, edges_AC)
    determinants = paz.algebra.dot(edges_AB, directions_cross_edges_AC)
    f = 1.0 / (determinants + EPSILON)
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
    depth = jp.where(hit_mask, depth, FARAWAY)
    return hit_mask, depth, u, v


def intersect_mesh(mesh, ray_origins, ray_directions):
    world_to_shape = jp.linalg.inv(mesh.transform)
    rays = (ray_origins, ray_directions)
    rays = paz.algebra.transform_rays(world_to_shape, *rays)
    return intersect_canonical_mesh(mesh.vertices, mesh.faces, *rays)
