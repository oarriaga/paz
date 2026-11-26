import jax.numpy as jp


def build_canonical_ring_normals(points):
    x_points, y_points, z_points = jp.split(points, 3, 1)
    zeros = jp.zeros((len(points), 1))
    ring_normals = jp.hstack([x_points, zeros, z_points])
    return ring_normals
