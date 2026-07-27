import jax.numpy as jp

import paz

from .intersect import intersect_mesh
from .patterns import interpolate_for_hits


def render_coordinates(shape, y_FOV, pose, mesh, chunk_size):
    origins, directions = paz.graphics.camera.build_rays(shape, y_FOV, pose)
    intersection = intersect_mesh(mesh, origins, directions, chunk_size)
    hit, _, u, v, face_index = intersection
    args = mesh.vertices, mesh.faces, face_index, u, v
    coordinates = interpolate_for_hits(*args) * hit[:, None]
    H, W = shape
    return jp.reshape(coordinates, (H, W, 3)), jp.reshape(hit, (H, W))
