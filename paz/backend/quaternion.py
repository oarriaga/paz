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


def quarternion_to_rotation_matrix(q):
    """Transforms quarternion into rotation vector

    # Arguments
        q: quarternion, Numpy array of shape ``[4]``

    # Returns
        Numpy array representing a rotation vector having a shape ``[3]``.
    """
    rotation_matrix = np.array([[1 - 2*(q[1]**2 + q[2]**2), 2*(q[0]*q[1] - q[3]*q[2]), 2*(q[3]*q[1] + q[0]*q[2])],
                                [2*(q[0]*q[1] + q[3]*q[2]), 1 - 2*(q[0]**2 + q[2]**2), 2*(q[1]*q[2] - q[3]*q[0])],
                                [2*(q[0]*q[2] - q[3]*q[1]), 2*(q[3]*q[0] + q[1]*q[2]), 1 - 2*(q[0]**2 + q[1]**2)]])

    return np.squeeze(rotation_matrix)


def quaternion_multiply(quaternion1, quaternion0):
    # Code taken from https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)