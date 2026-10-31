import jax.numpy as jp

import paz

from .intersect import intersect_mesh
from .patterns import interpolate_for_hits


def render_coordinates(shape, intrinsics, pose, mesh, face_chunk_size):
    rays = paz.graphics.camera.build_rays(shape, intrinsics, pose)
    intersection = intersect_mesh(mesh, *rays, face_chunk_size)
    hit, _, u, v, face_index = intersection
    args = mesh.vertices, mesh.faces, face_index, u, v
    coordinates = interpolate_for_hits(*args) * hit[:, None]
    H, W = shape
    return jp.reshape(coordinates, (H, W, 3)), jp.reshape(hit, (H, W))
