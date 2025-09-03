import jax.numpy as jp
import paz


def compute_colors(points3D, image):
    u, v = cylindrical_map(points3D)
    colors = paz.graphics.patterns.image.compute_image_colors(u, v, image)
    return colors


def cylindrical_map(points3D):
    x, y, z = jp.split(points3D, 3, axis=1)
    theta = jp.arctan2(x, z)
    raw_u = theta / (2 * jp.pi)
    u = 1 - (raw_u + 0.5)
    v = jp.remainder(y, 1)
    return u, v
