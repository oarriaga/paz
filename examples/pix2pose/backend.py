import numpy as np


def homogenous_quaternion_to_rotation_matrix(quaternion):
    # w0, q1, q2, q3 = quaternion
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


def inhomogenous_quaternion_to_rotation_matrix(q):
    """Transforms quaternion into a rotation matrix
    # Arguments
        q: quarternion, Numpy array of shape ``[4]``
    # Returns
        Numpy array representing a rotation vector having a shape ``[3]``.
    """
    # quaternion
    # q = q[::-1]
    r11 = 1 - (2 * (q[1]**2 + q[2]**2))
    r12 = 2 * (q[0] * q[1] - q[3] * q[2])
    r13 = 2 * (q[3] * q[1] + q[0] * q[2])

    r21 = 2 * (q[0] * q[1] + q[3] * q[2])
    r22 = 1 - (2 * (q[0]**2 + q[2]**2))
    r23 = 2 * (q[1] * q[2] - q[3] * q[0])

    r31 = 2 * (q[0] * q[2] - q[3] * q[1])
    r32 = 2 * (q[3] * q[0] + q[1] * q[2])
    r33 = 1 - (2 * (q[0]**2 + q[1]**2))

    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])

    return rotation_matrix
    # return np.squeeze(rotation_matrix)


def multiply_quaternions(quaternion_0, quaternion_1):
    """Multiplies two quaternions.

    # Reference:
        Code extracted from [here](https://stackoverflow.com/questions/
            39000758/how-to-multiply-two-quaternions-by-python-or-numpy)
    """
    x0, y0, z0, w0 = quaternion_0
    x1, y1, z1, w1 = quaternion_1
    x2 = +(x1 * w0) + (y1 * z0) - (z1 * y0) + (w1 * x0)
    y2 = -(x1 * z0) + (y1 * w0) + (z1 * x0) + (w1 * y0)
    z2 = +(x1 * y0) - (y1 * x0) + (z1 * w0) + (w1 * z0)
    w2 = -(x1 * x0) - (y1 * y0) - (z1 * z0) + (w1 * w0)
    return np.array([x2, y2, z2, w2])


# quaternion = (1 / np.sqrt(30)) * np.array([1, 2, 3, 4])
# theta = np.deg2rad(0)
# quaternion = np.array([1, 0, 0, 0])
# a = homogenous_quaternion_to_rotation_matrix(quaternion)
# quaternion = (1 / np.sqrt(30)) * np.array([2, 3, 4, 1])
# b = inhomogenous_quaternion_to_rotation_matrix(quaternion)
