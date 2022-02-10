import numpy as np

from paz.backend.boxes import extract_bounding_box_corners
from paz.backend.image import normalize_min_max
from paz.backend.groups import build_rotation_matrix_x
from paz.backend.groups import build_rotation_matrix_y
from paz.backend.groups import build_rotation_matrix_z
from paz.backend.groups import to_affine_matrix


def compute_vertices_colors(vertices):
    corner3D_min, corner3D_max = extract_bounding_box_corners(vertices)
    normalized_colors = normalize_min_max(vertices, corner3D_min, corner3D_max)
    colors = (255 * normalized_colors).astype('uint8')
    return colors


def sample_uniform(min_value, max_value):
    """Samples values inside segment [min_value, max_value)

    # Arguments
        segment_limits: List (2) containing min and max segment values.

    # Returns
        Float inside segment [min_value, max_value]
    """
    if min_value > max_value:
        raise ValueError('First value must be lower than second value')
    value = np.random.uniform(min_value, max_value)
    return value


def sample_inside_box3D(min_W, min_H, min_D, max_W, max_H, max_D):
    """ Samples points inside a 3D box defined by the
        width, height and depth limits.
                    ________
                   /       /|
                  /       / |
                 /       /  |
                /_______/   /
         |      |       |  /   /
       height   |       | / depth
         |      |_______|/   /

                --widht--

    # Arguments
        width_limits: List (2) with [min_value_width, max_value_width].
        height_limits: List (2) with [min_value_height, max_value_height].
        depth_limits: List (2) with [min_value_depth, max_value_depth].

    # Returns
        Array (3) of point inside the 3D box.
    """
    W = sample_uniform(min_W, max_W)
    H = sample_uniform(min_H, max_H)
    D = sample_uniform(min_D, max_D)
    box_point3D = np.array([W, H, D])
    return box_point3D


def sample_front_rotation_matrix(epsilon=0.1):
    x_angle = np.random.uniform((-np.pi / 2.0) + epsilon,
                                (np.pi / 2.0) - epsilon)
    y_angle = np.random.uniform((-np.pi / 2.0) + epsilon,
                                (np.pi / 2.0) - epsilon)
    z_angle = np.random.uniform(np.pi, -np.pi)

    x_matrix = build_rotation_matrix_x(x_angle)
    y_matrix = build_rotation_matrix_y(y_angle)
    z_matrix = build_rotation_matrix_z(z_angle)

    rotation_matrix = np.dot(z_matrix, np.dot(y_matrix, x_matrix))
    return rotation_matrix


def sample_affine_transform(min_corner, max_corner):
    min_W, min_H, min_D = min_corner
    max_W, max_H, max_D = max_corner
    translation = sample_inside_box3D(min_W, min_H, min_D, max_W, max_H, max_D)
    rotation_matrix = sample_front_rotation_matrix()
    affine_matrix = to_affine_matrix(rotation_matrix, translation)
    return affine_matrix
