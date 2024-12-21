import jax
import jax.numpy as jp

from ..math import safe_norm, divide, near_zero


def hat(so3_vector):
    """Isomorphism from R^3 to skew-symmetric form.

    # Arguments
        so3_vector: Numpy array with shape (3).

    # Return
        The skew symmetric representation of so3_vector.
    """
    return jp.array([[0.0, -so3_vector[2], so3_vector[1]],
                     [so3_vector[2], 0.0, -so3_vector[0]],
                     [-so3_vector[1], so3_vector[0], 0.0]])


def vee(so3_matrix):
    """Isomorphism from skew-symmetric matrix form to vector form.

    # Arguments
        so3_matrix: A 3x3 skew-symmetric matrix.

    # Returns
        The 3-vector corresponding to a so(3) matrix
    """
    return jp.array([so3_matrix[2, 1], so3_matrix[0, 2], so3_matrix[1, 0]])


def compute_rotation_angle(exponential_coordinates):
    """Computes rotation angle from exponential coordinates

    # Arguments
        exponential_coordinates: Array of shape (3).

    # Returns
        Float. The corresponding rotation angle.
    """
    # return jp.linalg.norm(exponential_coordinates)
    return safe_norm(exponential_coordinates)


def compute_rodriguez_formula(theta, omega_matrix):
    """Computes the matrix exponential of rotations using Rodrigues' formula

    # Arguments
        theta: scalar.
        omega_matrix: skew symmetric matrix.

    # Returns
        matrix exponential of rotations
    """
    first_order = jp.eye(3)
    odds_powers = jp.sin(theta) * omega_matrix
    even_powers = (1 - jp.cos(theta)) * jp.dot(omega_matrix, omega_matrix)
    return first_order + odds_powers + even_powers


def exp(matrix_so3):
    """Computes the matrix exponential of matrix in so(3)

    # Arguments
        matrix_so3: A 3x3 skew-symmetric matrix.

    # Returns
        Element of SO3 representing manifold projection of tangent matrix so3
    """
    omega = vee(matrix_so3)
    theta = compute_rotation_angle(omega)
    omega_matrix = divide(matrix_so3, theta)
    SO3 = compute_rodriguez_formula(theta, omega_matrix)
    return SO3


def rpy_to_SO3(coordinates):
    """Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.

    # Arguments
        coordinates: (3) float. Roll-pitch-yaw coordinates in order (x, y, z).

    # Returns
        Homogenous (3,3) rotation matrix.

    # Notes
        The roll-pitch-yaw axes are usually defined for a URDF as a
        rotation of ``r`` radians around the x-axis followed by a rotation of
        ``p`` radians around the y-axis followed by a rotation of ``y`` radians
        around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles, see [1].

    # References
        [Wikipedia](https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix)
    """
    coordinates = jp.array(coordinates)
    c3, c2, c1 = jp.cos(coordinates)
    s3, s2, s1 = jp.sin(coordinates)
    return jp.array([
        [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
        [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
        [-s2, c2 * s3, c2 * c3]])


def case_0_log(R):
    return jp.zeros((3, 3))


def case_1_log(SO3):
    choose_omega_0 = jp.logical_not(near_zero(1 + SO3[2, 2]))
    choose_omega_1 = jp.logical_not(near_zero(1 + SO3[1, 1]))
    omega_0 = (1.0 / jp.sqrt(2 * (1 + SO3[2, 2]))) * jp.array([SO3[0, 2], SO3[1, 2], 1 + SO3[2, 2]])
    omega_1 = (1.0 / jp.sqrt(2 * (1 + SO3[1, 1]))) * jp.array([SO3[0, 1], SO3[1, 1], 1 + SO3[2, 1]])
    omega_2 = (1.0 / jp.sqrt(2 * (1 + SO3[0, 0]))) * jp.array([1 + SO3[0, 0], SO3[1, 0], SO3[2, 0]])
    omega = jp.where(choose_omega_0, omega_0, jp.where(choose_omega_1, omega_1, omega_2))
    return omega


def case_2_log(SO3):
    arccos_input = (jp.trace(SO3) - 1) / 2.0
    theta = jp.arccos(arccos_input - 1e-6)
    return (theta / (2.0 * jp.sin(theta) + 1e-6)) * (SO3 - SO3.T)


def log(SO3_matrix):
    acosinput = (jp.trace(SO3_matrix) - 1) / 2.0
    choose_0 = acosinput >= 1
    choose_1 = acosinput <= -1
    return jp.where(
        choose_0,
        case_0_log(SO3_matrix),
        jp.where(
            choose_1,
            case_1_log(SO3_matrix),
            case_2_log(SO3_matrix)))


def sample(key):
    normal_matrix = jax.random.normal(key, (3, 3))
    orthonormal_matrix = jp.linalg.qr(normal_matrix)[0]
    negative_determinant = jp.linalg.det(orthonormal_matrix) < 0
    column_0 = orthonormal_matrix[:, 0:1]
    column_1 = orthonormal_matrix[:, 1:2]
    column_2 = orthonormal_matrix[:, 2:3]
    SO3_positive_det = jp.concatenate([column_0, column_1, column_2], axis=1)
    SO3_negative_det = jp.concatenate([column_1, column_0, column_2], axis=1)
    return jp.where(negative_determinant, SO3_negative_det, SO3_positive_det)


def rotation_z(angle):
    """Builds rotation matrix in Z axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (3, 3) rotation matrix in Z axis.
    """
    cos_angle = jp.cos(angle)
    sin_angle = jp.sin(angle)
    rotation_matrix_z = jp.array([[+cos_angle, -sin_angle, 0.0],
                                  [+sin_angle, +cos_angle, 0.0],
                                  [0.0, 0.0, 1.0]])
    return rotation_matrix_z


def rotation_x(angle):
    """Builds rotation matrix in X axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (3, 3) rotation matrix in X axis.
    """
    cos_angle = jp.cos(angle)
    sin_angle = jp.sin(angle)
    rotation_matrix_x = jp.array([[1.0, 0.0, 0.0],
                                  [0.0, +cos_angle, -sin_angle],
                                  [0.0, +sin_angle, +cos_angle]])
    return rotation_matrix_x


def rotation_y(angle):
    """Builds affine rotation matrix in Y axis.

    # Arguments
        angle: Float. Angle in radians.

    # Return
        Array (3, 3) rotation matrix in Y axis.
    """
    cos_angle = jp.cos(angle)
    sin_angle = jp.sin(angle)
    rotation_matrix_y = jp.array([[+cos_angle, 0.0, +sin_angle],
                                  [0.0, 1.0, 0.0],
                                  [-sin_angle, 0.0, +cos_angle]])
    return rotation_matrix_y