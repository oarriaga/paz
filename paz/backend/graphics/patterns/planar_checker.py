import jax.numpy as jp

from paz.backend.graphics.patterns.checker import compute_checker_colors


def planar_map(points3D):
    x, y, z = jp.split(points3D, 3, axis=1)
    u = jp.remainder(x, 1)
    v = jp.remainder(z, 1)
    return u, v


def compute_colors(points3D, color_A, color_B):
    u, v = planar_map(points3D)
    colors = compute_checker_colors(u, v, color_A, color_B, 2, 2)
    return colors
