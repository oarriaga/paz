import numpy as np

from paz.backend.boxes import extract_bounding_box_corners
from paz.backend.image import normalize_min_max
from paz.backend.image.draw import draw_cube
from paz.backend.keypoints import points3D_to_RGB
from paz.backend.keypoints import project_to_image

from paz.backend.groups import build_rotation_matrix_x
from paz.backend.groups import build_rotation_matrix_y
from paz.backend.groups import build_rotation_matrix_z
from paz.backend.groups import to_affine_matrix
from paz.backend.groups import quaternion_to_rotation_matrix


def draw_mask(image, points2D, points3D, object_sizes):
    colors = points3D_to_RGB(points3D, object_sizes)
    image = draw_points2D(image, points2D, colors)
    return image


# TODO change to processor
def draw_masks(image, points, object_sizes):
    for points2D, points3D in points:
        colors = points3D_to_RGB(points3D, object_sizes)
        image = draw_points2D(image, points2D, colors)
    return image


def draw_points2D(image, points2D, colors):
    """Draws a pixel for all points2D in UV space using only numpy.

    # Arguments
        image: Array (H, W).
        keypoints: Array (num_points, U, V). Keypoints in image space
        colors: Array (num_points, 3). Colors in RGB space.

    # Returns
        Array with drawn points.
    """
    points2D = points2D.astype(int)
    U = points2D[:, 0]
    V = points2D[:, 1]
    image[V, U, :] = colors
    return image


def draw_pose6D(image, pose6D, cube_points3D, camera_intrinsics, thickness=2):
    """Draws pose6D by projecting cube3D to image space with camera intrinsics.

    # Arguments
        image: Array (H, W, 3)
        pose6D: paz message Pose6D with quaternion and translation values.
        cube3D: Array (8, 3). Cube 3D points in object frame.
        camera_intrinsics: Array of shape (3, 3). Diagonal elements represent
            focal lenghts and last column the image center translation.

    # Returns
        Original image array (H, W, 3) with drawn cube points.
    """
    quaternion, translation = pose6D.quaternion, pose6D.translation
    rotation = quaternion_to_rotation_matrix(quaternion)
    rotation = np.squeeze(rotation, axis=2)
    cube_points2D = project_to_image(
        rotation, translation, cube_points3D, camera_intrinsics)
    cube_points2D = cube_points2D.astype(np.int32)
    image = draw_cube(image, cube_points2D, thickness=thickness)
    return image


def draw_poses6D(image, poses6D, cube_points3D,
                 camera_intrinsics, thickness=2):
    """Draws pose6D by projecting cube3D to image space with camera intrinsics.

    # Arguments
        image: Array (H, W, 3)
        pose6D: List paz messages Pose6D with quaternions and translations.
        cube3D: Array (8, 3). Cube 3D points in object frame.
        camera_intrinsics: Array of shape (3, 3). Diagonal elements represent
            focal lenghts and last column the image center translation.

    # Returns
        Original image array (H, W, 3) with drawn cube points for all poses6D.
    """
    for pose6D in poses6D:
        image = draw_pose6D(image, pose6D, cube_points3D,
                            camera_intrinsics, thickness)
    return image


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
