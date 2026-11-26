import jax.numpy as jp
import paz


def build_upper_cap_normals(points):
    num_rays = len(points)
    zeros, ones = jp.zeros((num_rays, 1)), jp.ones((num_rays, 1))
    upper_cap_normals = jp.hstack([zeros, ones, zeros])
    return upper_cap_normals


def build_lower_cap_normals(points):
    num_rays = len(points)
    zeros, ones = jp.zeros((num_rays, 1)), jp.ones((num_rays, 1))
    lower_cap_normals = jp.hstack([zeros, -ones, zeros])
    return lower_cap_normals


def intersect_infinite_plane(ray_origins, ray_directions, y_translation):
    y_origins, y_directions = ray_origins[:, 1:2], ray_directions[:, 1:2]
    # A ray along the plane meets it far away, on the side it started from.
    return paz.algebra.divide_far(y_translation - y_origins, y_directions)
