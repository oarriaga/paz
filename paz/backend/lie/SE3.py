import jax
import jax.numpy as jp

from . import SO3
from ..math import divide, near_zero, to_column_vector


def get_rotation_matrix(affine_matrix):
    """Get rotation matrix from affine matrix.

    # Arguments
        affine_matrix: Numpy array of shape (4, 4).

    # Return
        Numpy array of shape (3, 3). Rotation matrix.
    """
    rotation_matrix = affine_matrix[0:3, 0:3]
    return rotation_matrix


def get_position_vector(affine_matrix):
    """Get translation vector from affine matrix.

    # Arguments
        affine_matrix: Numpy array of shape (4, 4).

    # Return
        Numpy array of shape (4). Position (translation) vector.
    """
    translation_vector = affine_matrix[0:3, 3]
    return translation_vector


def split(affine_matrix):
    """Splits affine matrix into a rotation matrix and a position vector.

    # Arguments
        affine_matrix: Numpy array of shape (4, 4).

    # Returns
        Rotation matrix (3, 3) and position vector (3).
    """
    rotation_matrix = get_rotation_matrix(affine_matrix)
    position_vector = get_position_vector(affine_matrix)
    return rotation_matrix, position_vector


def get_angular_velocity(se3_vector):
    """Gets angular component of an element of se(3)

    # Arguments
        se3_vector: Array of shape (6)

    # Returns
        Array of shape (3)
    """
    return se3_vector[0:3]


def get_linear_velocity(se3_vector):
    """Gets linear component of an element of se(3)

    # Arguments
        se3_vector: Array of shape (6)

    # Returns
        Array of shape (3)
    """
    return se3_vector[3:6]


def hat(se3_vector):
    """Transform se(3) vector to 4x4 matrix form.

    # Arguments
        se3_vector: Array (6) representing element of se(3).

    # Returns
        The 4x4 matrix representation of of se(3) vector.
    """
    angular_velocity = get_angular_velocity(se3_vector)
    matrix_so3 = SO3.hat(angular_velocity)
    linear_velocity = get_linear_velocity(se3_vector)
    linear_velocity = jp.reshape(linear_velocity, (3, 1))
    upper_se3 = jp.concatenate([matrix_so3, linear_velocity], axis=1)
    lower_se3 = jp.zeros((1, 4))
    matrix_se3 = jp.concatenate([upper_se3, lower_se3], axis=0)
    return matrix_se3


def to_affine_matrix(rotation_matrix, translation):
    """Builds affine matrix from rotation matrix and translation vector.

    # Arguments
        rotation_matrix: Array (3, 3). Representing a rotation matrix.
        translation: Array (3). Translation vector.

    # Returns
        Array (4, 4) representing an affine matrix.
    """
    translation = translation.reshape(3, 1)
    affine_top = jp.concatenate([rotation_matrix, translation], axis=1)
    affine_row = jp.array([[0.0, 0.0, 0.0, 1.0]])
    affine_matrix = jp.concatenate([affine_top, affine_row], axis=0)
    return affine_matrix


def exp_position(theta, omega_matrix, position_vector):
    first_order = jp.eye(3) * theta
    odds_powers = (1 - jp.cos(theta)) * omega_matrix
    even_powers = (theta - jp.sin(theta)) * jp.dot(omega_matrix, omega_matrix)
    exponential_position = first_order + odds_powers + even_powers
    return divide(jp.dot(exponential_position, position_vector), theta)


def exp(matrix_se3):
    """Computes matrix exponential mapping elements of se(3) to SE(3).

    # Arguments
        matrix_se3: Array (4x4) representing an element of se(3)

    # Returns
        matrix_SE3: Array (4x4) representing an element of SE(3)
    """
    matrix_so3, position = split(matrix_se3)
    omega = SO3.vee(matrix_so3)
    theta = SO3.compute_rotation_angle(omega)
    omega_matrix = divide(matrix_so3, theta)
    rotation = SO3.exp(matrix_so3)
    position = jp.reshape(position, (3, 1))
    position_SE3 = exp_position(theta, omega_matrix, position)
    # safe for auto-differentiation
    position_SE3 = jp.where(near_zero(theta), position, position_SE3)
    affine_matrix = jp.concatenate([rotation, position_SE3], axis=1)
    affine_row = jp.array([[0.0, 0.0, 0.0, 1.0]])
    affine_matrix = jp.concatenate([affine_matrix, affine_row], axis=0)
    return affine_matrix


def xyz_rpy_to_SE3(position, rotation):
    rotation = SO3.rpy_to_SO3(rotation)
    return to_affine_matrix(rotation, position)


def translation(translation_vector):
    """Builds affine translation matrix

    # Arguments
        angle: Array (3) having [x, y, z] coordinates.

    # Return
        Array (4, 4) translation matrix.
    """
    x, y, z = translation_vector
    return jp.array([[1.0, 0.0, 0.0, x],
                     [0.0, 1.0, 0.0, y],
                     [0.0, 0.0, 1.0, z],
                     [0.0, 0.0, 0.0, 1.0]])


def invert(SE3_matrix):
    """Inverts a homogeneous transformation matrix

    # Argument
        affine_matrix: Numpy array of shape (4, 4).

    # Return
        Numpy array of shape (4, 4) having the inverse of the affine matrix.
    """
    rotation_matrix, position_vector = split(SE3_matrix)
    rotation_matrix_transposed = rotation_matrix.T
    position_vector = jp.reshape(position_vector, (3, 1))
    position_inverse = -jp.dot(rotation_matrix_transposed, position_vector)
    inverse = jp.concatenate([rotation_matrix_transposed, position_inverse], 1)
    affine_row = jp.array([[0.0, 0.0, 0.0, 1.0]])
    inverse = jp.concatenate([inverse, affine_row], axis=0)
    return inverse


def Ad(affine_matrix):
    """Computes adjoint representation of a homogeneous transformation matrix

    # Arguments
        affine_matrix: A homogeneous transformation matrix

    # Return
        The 6x6 adjoint representation [AdT] where T is the input affine matrix
    """
    R, p = split(affine_matrix)
    upper_3x6_adjoint = jp.concatenate([R, jp.zeros((3, 3))], axis=1)
    pxR = jp.dot(SO3.hat(p), R)
    lower_3x6_adjoint = jp.concatenate([pxR, R], axis=1)
    adjoint = jp.concatenate([upper_3x6_adjoint, lower_3x6_adjoint], axis=0)
    return adjoint


def ad(twist):
    """Calculate the 6x6 matrix [adV] of the given 6-vector.
    Used to calculate the Lie bracket [V1, V2] = [adV1]V2

    # Arguments
        A 6-vector spatial velocity

    # Returns
        The corresponding 6x6 matrix [adV] where V is the inputted twist
    """
    omega_matrix_w = SO3.hat(twist[:3])
    upper_3x6 = jp.concatenate([omega_matrix_w, jp.zeros((3, 3))], axis=1)
    omega_matrix_v = SO3.hat(twist[3:])
    lower_3x6 = jp.concatenate([omega_matrix_v, omega_matrix_w], axis=1)
    cross_operator = jp.concatenate([upper_3x6, lower_3x6], axis=0)
    return cross_operator


def case_log_0(SE3_matrix):
    return jp.r_[jp.c_[jp.zeros((3, 3)), [SE3_matrix[0, 3], SE3_matrix[1, 3], SE3_matrix[2, 3]]], [[0, 0, 0, 0]]]

    omega = jp.zeros((3, 3))
    se3_matrix = jp.concatenate([omega, SE3_matrix[:3, 3]], axis=1)
    se3_matrix = jp.concatenate([se3_matrix, jp.zeros((1, 4))])
    return se3_matrix


def case_log_1(SE3_matrix):
    rotation, position = split(SE3_matrix)
    omega_matrix = SO3.log(rotation) # return [w]*theta
    theta = jp.arccos((jp.trace(rotation) - 1) / 2.0)
    omega = omega_matrix / theta

    term_0 = (1.0 / theta) * jp.eye(3)
    term_1 = - 0.5 * omega
    cotangent = 1.0 / jp.tan(0.5 * theta)
    term_2 = ((1.0 / theta) - (0.5 * cotangent)) * (omega @ omega)
    G_inverse = term_0 + term_1 + term_2
    v = G_inverse @ to_column_vector(position)
    se3_matrix = jp.concatenate([omega_matrix, v * theta], axis=1)
    se3_matrix = jp.concatenate([se3_matrix, jp.zeros((1, 4))])
    return se3_matrix


def log(SE3_matrix):
    rotation, position = split(SE3_matrix)
    omega = SO3.log(rotation)
    choose_0 = jp.allclose(omega, jp.zeros((3, 3)))
    return jp.where(choose_0, case_log_0(SE3_matrix), case_log_1(SE3_matrix))


def vee(se3_matrix):
    """Transforms se3 4x4 matrix to it's vector representation
    """
    so3_matrix, se3_position = split(se3_matrix)
    so3_vector = SO3.vee(so3_matrix)
    return jp.concatenate([so3_vector, se3_position])


def sample(key, min_value, max_value):
    key_SO3, key_SE3 = jax.random.split(key)
    rotation = SO3.sample(key_SO3)
    position = jax.random.uniform(key, (3,), float, min_value, max_value)
    return to_affine_matrix(rotation, position)


def scaling(scaling_vector):
    """Builds scaling translation matrix

    # Arguments
        angle: Array (3) having [x, y, z] scaling values.

    # Return
        Array (4, 4) scale matrix.
    """
    x, y, z = scaling_vector
    return jp.array([[x, 0.0, 0.0, 0.0],
                     [0.0, y, 0.0, 0.0],
                     [0.0, 0.0, z, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])
