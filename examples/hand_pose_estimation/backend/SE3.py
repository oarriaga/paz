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


def scalar_to_vector(scalar):
    return [0, 0, scalar]


def build_translation_matrix_SE3(translation_vector):
    """ Build a translation matrix from translation vector.

    # Arguments
        translation_vector: list of length 1 or 3.

    # Returns
        transformation_matrix: Numpy array of size (1, 4, 4).
    """
    if len(translation_vector) == 1:  # Make it into another function
        translation_vector = scalar_to_vector(translation_vector)
    transformation_matrix = np.array([[1, 0, 0, translation_vector[0]],
                                      [0, 1, 0, translation_vector[1]],
                                      [0, 0, 1, translation_vector[2]],
                                      [0, 0, 0, 1]])
    transformation_matrix = np.expand_dims(transformation_matrix, 0)
    return transformation_matrix


def build_affine_matrix(matrix):
    """ Build a (4, 4) affine matrix provided a matrix of size (3, 3).

    # Arguments
        matrix: numpy array of shape (3, 3).

    # Returns
        affine_matrix: Numpy array of size (4, 4).
    """
    translation_vector = np.array([[0], [0], [0]])
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
    sine_value = np.cos(angle)
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
    sine_value = np.cos(angle)
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
    sine_value = np.cos(angle)
    rotation_matrix_z = np.array([[cosine_value, sine_value, 0.0],
                                  [-sine_value, cosine_value, 0.0],
                                  [0.0, 0.0, 1.0]])
    return rotation_matrix_z


def get_axis_coordinates(axis_angles, theta, is_normalized):
    """ Calculate axis coordinates.

    # Arguments:
        axis_angles: Numpy array of shape (batch_size, 3).
        theta: Float value.
        is_normalized: boolean value.

    # Returns:
        ux, uy, uz: Float values.
    """
    ux = axis_angles[:, 0]
    uy = axis_angles[:, 1]
    uz = axis_angles[:, 2]

    if not is_normalized:
        normalization_factor = 1.0 / theta
        ux = ux * normalization_factor
        uy = uy * normalization_factor
        uz = uz * normalization_factor
    return ux, uy, uz


def get_rotation_matrix_elements(axis_coordinates, theta):
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

    m00 = np.cos(theta) + ((x ** 2) * (1.0 - np.cos(theta)))
    m11 = np.cos(theta) + ((y ** 2) * (1.0 - np.cos(theta)))
    m22 = np.cos(theta) + ((z ** 2) * (1.0 - np.cos(theta)))

    m01 = (x * y * (1.0 - np.cos(theta))) - (z * np.sin(theta))
    m02 = (x * z * (1.0 - np.cos(theta))) + (y * np.sin(theta))
    m10 = (y * x * (1.0 - np.cos(theta))) + (z * np.sin(theta))
    m12 = (y * z * (1.0 - np.cos(theta))) - (x * np.sin(theta))
    m20 = (z * x * (1.0 - np.cos(theta))) - (y * np.sin(theta))
    m21 = (z * y * (1.0 - np.cos(theta))) + (x * np.sin(theta))

    matrix = np.array([[m00, m01, m02],
                       [m10, m11, m12],
                       [m20, m21, m22]])
    return matrix


def rotation_from_axis_angles(axis_angles, is_normalized=False):
    """ Get Rotation matrix from axis angles.

    # Arguments
        axis_angles: list of length (3).
        is_normalized: boolean value.

    # Returns
        rotation-matrix: numpy array of size (3, 3).
    """
    theta = np.linalg.norm(axis_angles)
    ux, uy, uz = get_axis_coordinates(axis_angles, theta, is_normalized)
    rotation_matrix = get_rotation_matrix_elements([ux, uy, uz], theta)
    return rotation_matrix
