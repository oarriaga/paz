import numpy as np


def to_homogeneous_coordinates(vector):
    """ Homogenize the vector : Appending 1 to the vector.

    # Arguments
        keypoints: Numpy array with any shape.

    # Returns
        vector: Numpy array.
    """
    vector = np.append(vector, 1)
    return vector


def build_translation_matrix_SE3(translation_vector):
    """ Build a translation matrix from translation vector.

    # Arguments
        translation_vector: list of length 1 or 3.

    # Returns
        transformation_matrix: Numpy array of size (1, 4, 4).
    """
    if len(translation_vector) == 1:
        translation_vector = [0, 0, translation_vector]
    transformation_matrix = np.array([[1, 0, 0, translation_vector[0]],
                                      [0, 1, 0, translation_vector[1]],
                                      [0, 0, 1, translation_vector[2]],
                                      [0, 0, 0, 1]])
    return transformation_matrix


def build_affine_matrix(matrix, translation_vector=None):
    """ Build a (4, 4) affine matrix provided a matrix of size (3, 3).

    # Arguments
        matrix: numpy array of shape (3, 3).

    # Returns
        affine_matrix: Numpy array of size (4, 4).
    """
    if translation_vector is None:
        translation_vector = np.array([[0], [0], [0]])

    if len(translation_vector) == 1:
        translation_vector = [0, 0, translation_vector]

    affine_matrix = np.hstack([matrix, translation_vector])
    affine_matrix = np.vstack((affine_matrix, [0, 0, 0, 1]))
    return affine_matrix


def build_rotation_matrix_x(angle):
    """Build a (3, 3) rotation matrix along x-axis.

    # Arguments
        angle: float value of range [0, 360].

    # Returns
        rotation_matrix_x: Numpy array of size (3, 3).
    """
    cosine_value = np.cos(angle)
    sine_value = np.sin(angle)
    rotation_matrix_x = np.array([[1.0, 0.0, 0.0],
                                  [0.0, cosine_value, sine_value],
                                  [0.0, -sine_value, cosine_value]])
    return rotation_matrix_x


def build_rotation_matrix_y(angle):
    """Build a (3, 3) rotation matrix along y-axis.

    # Arguments
        angle: float value of range [0, 360].

    # Returns
        rotation_matrix_y: Numpy array of size (3, 3).
    """
    cosine_value = np.cos(angle)
    sine_value = np.sin(angle)
    rotation_matrix_y = np.array([[cosine_value, 0.0, -sine_value],
                                  [0.0, 1.0, 0.0],
                                  [sine_value, 0.0, cosine_value]])
    return rotation_matrix_y


def build_rotation_matrix_z(angle):
    """ Build a (3, 3) rotation matrix along z-axis.

    # Arguments
        angle: float value of range [0, 360].

    # Returns
        rotation_matrix_z: Numpy array of size (3, 3).
    """
    cosine_value = np.cos(angle)
    sine_value = np.sin(angle)
    rotation_matrix_z = np.array([[cosine_value, sine_value, 0.0],
                                  [-sine_value, cosine_value, 0.0],
                                  [0.0, 0.0, 1.0]])
    return rotation_matrix_z


def normalize_axis_coordinates(axis_angles, theta):
    normalization_factor = 1.0 / theta
    axis_coordinates_x = axis_angles[0] * normalization_factor
    axis_coordinates_y = axis_angles[1] * normalization_factor
    axis_coordinates_z = axis_angles[2] * normalization_factor
    axis_angles = (axis_coordinates_x, axis_coordinates_y, axis_coordinates_z)
    return axis_angles


def get_rotation_matrix(axis_coordinates, theta):
    """ Calculate Rotation matrix.

    # Arguments
        axis_coordinates: List of length (3).
        theta: Float value.

    # Returns:
        matrix: Numpy array of size (3, 3).
    """
    x = axis_coordinates[0]
    y = axis_coordinates[1]
    z = axis_coordinates[2]

    sine_theta = np.sin(theta)
    cosine_theta = np.cos(theta)

    r11 = cosine_theta + ((x ** 2) * (1.0 - cosine_theta))
    r22 = cosine_theta + ((y ** 2) * (1.0 - cosine_theta))
    r33 = cosine_theta + ((z ** 2) * (1.0 - cosine_theta))

    r12 = (x * y * (1.0 - cosine_theta)) - (z * sine_theta)
    r13 = (x * z * (1.0 - cosine_theta)) + (y * sine_theta)
    r21 = (y * x * (1.0 - cosine_theta)) + (z * sine_theta)
    r23 = (y * z * (1.0 - cosine_theta)) - (x * sine_theta)
    r31 = (z * x * (1.0 - cosine_theta)) - (y * sine_theta)
    r32 = (z * y * (1.0 - cosine_theta)) + (x * sine_theta)

    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])

    return rotation_matrix


def rotation_from_axis_angles(axis_angles, is_normalized=False):
    """ Get Rotation matrix from axis angles.

    # Arguments
        axis_angles: list of length (3).
        is_normalized: boolean value.

    # Returns
        rotation-matrix: numpy array of size (3, 3).
    """
    theta = np.linalg.norm(axis_angles)
    if not is_normalized:
        axis_angles = normalize_axis_coordinates(axis_angles, theta)
    rotation_matrix = get_rotation_matrix(axis_angles, theta)
    return rotation_matrix
