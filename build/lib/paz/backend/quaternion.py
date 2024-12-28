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
