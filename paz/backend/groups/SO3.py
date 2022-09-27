import numpy as np

# TODO: rmeove cv2 import by computing explicitly rodriguez formula
import cv2


def rotation_vector_to_rotation_matrix(rotation_vector):
    """Transforms rotation vector (axis-angle) form to rotation matrix.

    # Arguments
        rotation_vector: Array (3). Rotation vector in axis-angle form.

    # Returns
        Array (3, 3) rotation matrix.
    """
    rotation_matrix = np.eye(3)
    cv2.Rodrigues(rotation_vector, rotation_matrix)
    return rotation_matrix


def build_rotation_matrix_z(angle):
    """Builds rotation matrix in Z axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (3, 3) rotation matrix in Z axis.
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix_z = np.array([[+cos_angle, -sin_angle, 0.0],
                                  [+sin_angle, +cos_angle, 0.0],
                                  [0.0, 0.0, 1.0]])
    return rotation_matrix_z


def build_rotation_matrix_x(angle):
    """Builds rotation matrix in X axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (3, 3) rotation matrix in Z axis.
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix_x = np.array([[1.0, 0.0, 0.0],
                                  [0.0, +cos_angle, -sin_angle],
                                  [0.0, +sin_angle, +cos_angle]])
    return rotation_matrix_x


def build_rotation_matrix_y(angle):
    """Builds rotation matrix in Y axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (3, 3) rotation matrix in Z axis.
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix_y = np.array([[+cos_angle, 0.0, +sin_angle],
                                  [0.0, 1.0, 0.0],
                                  [-sin_angle, 0.0, +cos_angle]])
    return rotation_matrix_y


def compute_norm_SO3(rotation_mesh, rotation):
    """Computes norm between SO3 elements.

    # Arguments
        rotation_mesh: Array (3, 3), rotation matrix.
        rotation: Array (3, 3), rotation matrix.

    # Returns
        Scalar representing the distance between both rotation matrices.
    """
    difference = np.dot(np.linalg.inv(rotation), rotation_mesh) - np.eye(3)
    distance = np.linalg.norm(difference, ord='fro')
    return distance


def calculate_canonical_rotation(rotation_mesh, rotations):
    """Returns the rotation matrix closest to rotation mesh.

    # Arguments
        rotation_mesh: Array (3, 3), rotation matrix.
        rotations: List of array of (3, 3), rotation matrices.

    # Returns
        Element of list closest to rotation mesh.
    """
    norms = [compute_norm_SO3(rotation_mesh, R) for R in rotations]
    closest_rotation_arg = np.argmin(norms)
    closest_rotation = rotations[closest_rotation_arg]
    canonical_rotation = np.linalg.inv(closest_rotation)
    return canonical_rotation


def rotation_matrix_to_axis_angle(rotation_matrix):
    """Transforms rotation matrix to axis angle.

    # Arguments
        Rotation matrix [3, 3].

    # Returns
        axis_angle: Array containing axis angle represent [wx, wy, wz, theta].
    """
    cos_theta = (np.trace(rotation_matrix) - 1.0) / 2.0
    angle = np.arccos(cos_theta)
    axis = np.array([rotation_matrix[2, 1] - rotation_matrix[1, 2],
                     rotation_matrix[0, 2] - rotation_matrix[2, 0],
                     rotation_matrix[1, 0] - rotation_matrix[0, 1]])
    axis = axis / np.linalg.norm(axis)
    axis_angle = np.hstack([axis, angle])
    return axis_angle


def rotation_matrix_to_compact_axis_angle(matrix):
    """Transforms rotation matrix to compact axis angle.

    # Arguments
        Rotation matrix [3, 3].

    # Returns
        compact axis_angle
    """
    axis_angle = rotation_matrix_to_axis_angle(matrix)
    axis = axis_angle[:3]
    angle = axis_angle[3]
    compact_axis_angle = axis * angle
    return compact_axis_angle
