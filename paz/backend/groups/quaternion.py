import numpy as np


def rotation_vector_to_quaternion(rotation_vector):
    """Transforms rotation vector into quaternion.

    # Arguments
        rotation_vector: Numpy array of shape ``[3]``.

    # Returns
        Numpy array representing a quaternion having a shape ``[4]``.
    """
    theta = np.linalg.norm(rotation_vector)
    rotation_axis = rotation_vector / theta
    half_theta = 0.5 * theta
    norm = np.sin(half_theta)
    quaternion = np.array([
        norm * rotation_axis[0],
        norm * rotation_axis[1],
        norm * rotation_axis[2],
        np.cos(half_theta)])
    return quaternion


def homogenous_quaternion_to_rotation_matrix(quaternion):
    """Transforms quaternion to rotation matrix.

    # Arguments
        quaternion: Array containing quaternion value [q1, q2, q3, w0].

    # Returns
        Rotation matrix [3, 3].

    # Note
        If quaternion is not a unit quaternion the rotation matrix is not
        unitary but still orthogonal i.e. the outputted rotation matrix is
        a scalar multiple of a rotation matrix.
    """
    q1, q2, q3, w0 = quaternion

    r11 = w0**2 + q1**2 - q2**2 - q3**2
    r12 = 2 * ((q1 * q2) - (w0 * q3))
    r13 = 2 * ((w0 * q2) + (q1 * q3))

    r21 = 2 * ((w0 * q3) + (q1 * q2))
    r22 = w0**2 - q1**2 + q2**2 - q3**2
    r23 = 2 * ((q2 * q3) - (w0 * q1))

    r31 = 2 * ((q1 * q3) - (w0 * q2))
    r32 = 2 * ((w0 * q1) + (q2 * q3))
    r33 = w0**2 - q1**2 - q2**2 + q3**2

    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])
    return rotation_matrix


def quaternion_to_rotation_matrix(quaternion):
    """Transforms quaternion to rotation matrix.

    # Arguments
        quaternion: Array containing quaternion value [q1, q2, q3, w0].

    # Returns
        Rotation matrix [3, 3].

    # Note
        "If the quaternion "is not a unit quaternion then the homogeneous form
        is still a scalar multiple of a rotation matrix, while the
        inhomogeneous form is in general no longer an orthogonal matrix.
        This is why in numerical work the homogeneous form is to be preferred
        if distortion is to be avoided." [wikipedia](https://en.wikipedia.org/
            wiki/Conversion_between_quaternions_and_Euler_angles)
    """
    matrix = homogenous_quaternion_to_rotation_matrix(quaternion)
    return matrix


def rotation_matrix_to_quaternion(rotation_matrix):
    """Transforms rotation matrix to quaternion.

    # Arguments
        Rotation matrix [3, 3].

    # Returns
        quaternion: Array containing quaternion value [q1, q2, q3, w0].
    """
    rotation_matrix = rotation_matrix[:3, :3]
    trace = np.trace(rotation_matrix)
    w0 = np.sqrt(1.0 + trace) / 2
    q1 = 0.25 * (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / w0
    q2 = 0.25 * (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / w0
    q3 = 0.25 * (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / w0
    quaternion = np.array([q1, q2, q3, w0])
    return quaternion


def get_quaternion_conjugate(quaternion):
    """Estimate conjugate of a quaternion.

    # Arguments
        quaternion: Array containing quaternion value [q1, q2, q3, w0].

    # Returns
        quaternion: Array containing quaternion value [-q1, -q2, -q3, w0].
    """
    q1, q2, q3, w0 = quaternion
    return np.array([-q1, -q2, -q3, w0])


def quaternions_to_rotation_matrices(quaternions):
    """Transform quaternion vectors to rotation matrix vector.

    # Arguments
        quaternions [N, 4].

    # Returns
        Rotated matrices [N, 3, 3]
    """
    rotation_matrices = []
    for quaternion in quaternions:
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        rotation_matrices.append(rotation_matrix)
    return np.array(rotation_matrices)
