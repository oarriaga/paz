import math

import pytest

import jax
import jax.numpy as jp
from paz import SE3


@pytest.fixture
def se3_vector_A():
    return jp.array([1, 2, 3, 4, 5, 6])


@pytest.fixture
def so3_vector_A():
    return jp.array([1, 2, 3])


@pytest.fixture
def e3_vector_A():
    return jp.array([4, 5, 6])


@pytest.fixture
def se3_matrix_A():
    return jp.array([[0, -3, 2, 4], [3, 0, -1, 5], [-2, 1, 0, 6], [0, 0, 0, 0]])


@pytest.fixture
def SE3_matrix_B():
    return jp.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 3], [0, 0, 0, 1]])


@pytest.fixture
def SE3_matrix_B_inverse():
    return jp.array([[1, 0, 0, 0], [0, 0, 1, -3], [0, -1, 0, 0], [0, 0, 0, 1]])


@pytest.fixture
def SO3_matrix_B():
    return jp.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


@pytest.fixture
def E3_vector_B():
    return jp.array([0, 0, 3])


@pytest.fixture
def se3_matrix_B():
    return jp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.57079633, 2.35619449],
            [0.0, 1.57079633, 0.0, 2.35619449],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )


@pytest.fixture
def Ad_B():
    return jp.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 3, 1, 0, 0],
            [3, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 1, 0],
        ]
    )


@pytest.fixture
def ad_A():
    return jp.array(
        [
            [0, -3, 2, 0, 0, 0],
            [3, 0, -1, 0, 0, 0],
            [-2, 1, 0, 0, 0, 0],
            [0, -6, 5, 0, -3, 2],
            [6, 0, -4, 3, 0, -1],
            [-5, 4, 0, -2, 1, 0],
        ]
    )


@pytest.fixture
def translation_matrix_A():
    return jp.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])


@pytest.fixture
def radians_vector():
    """in the order of roll-pitch-yaw"""
    return jp.array([math.pi / 4, math.pi / 6, math.pi / 3])


@pytest.fixture
def SE3_matrix_from_radians():
    return jp.array(
        [
            [0.43301266, -0.43559578, 0.78914917, 1],
            [0.75, 0.6597396, -0.04736713, 2],
            [-0.5, 0.6123724, 0.6123724, 3],
            [0, 0, 0, 1],
        ]
    )


@pytest.fixture
def exp_position_input():
    theta = 1.5707964
    omega_matrix = jp.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    position = jp.array([[0], [2.3561945], [2.3561945]])
    return theta, omega_matrix, position


@pytest.fixture
def sample_key():
    seed = 123
    return jax.random.PRNGKey(seed)


def test_get_position_vector(SE3_matrix_B, E3_vector_B):
    assert jp.all(SE3.get_position_vector(SE3_matrix_B) == E3_vector_B)


def test_get_rotation_matrix(SE3_matrix_B, SO3_matrix_B):
    assert jp.all(SE3.get_rotation_matrix(SE3_matrix_B) == SO3_matrix_B)


def test_split(SE3_matrix_B, SO3_matrix_B, E3_vector_B):
    rotation, position = SE3.split(SE3_matrix_B)
    assert jp.all(rotation == SO3_matrix_B)
    assert jp.all(position == E3_vector_B)


def test_hat(se3_vector_A, se3_matrix_A):
    assert jp.all(SE3.hat(se3_vector_A) == se3_matrix_A)


def test_exp(SE3_matrix_B, se3_matrix_B):
    assert jp.allclose(SE3_matrix_B, SE3.exp(se3_matrix_B))


def test_to_affine_matrix(SE3_matrix_B, SO3_matrix_B, E3_vector_B):
    affine_matrix = SE3.to_affine_matrix(SO3_matrix_B, E3_vector_B)
    assert jp.all(affine_matrix == SE3_matrix_B)


def test_get_angular_velocity(se3_vector_A, so3_vector_A):
    angular_velocity = SE3.get_angular_velocity(se3_vector_A)
    assert jp.all(angular_velocity == so3_vector_A)


def test_get_linear_velocity(se3_vector_A, e3_vector_A):
    linear_velocity = SE3.get_linear_velocity(se3_vector_A)
    assert jp.all(linear_velocity == e3_vector_A)


def test_invert(SE3_matrix_B, SE3_matrix_B_inverse):
    assert jp.allclose(SE3.invert(SE3_matrix_B), SE3_matrix_B_inverse)


def test_Ad(SE3_matrix_B, Ad_B):
    assert jp.allclose(SE3.Ad(SE3_matrix_B), Ad_B)


def test_ad(se3_vector_A, ad_A):
    assert jp.allclose(SE3.ad(se3_vector_A), ad_A)


def test_vee(se3_vector_A, se3_matrix_A):
    assert jp.all(SE3.vee(se3_matrix_A) == se3_vector_A)


def test_log(SE3_matrix_B, se3_matrix_B):
    assert jp.allclose(se3_matrix_B, SE3.log(SE3_matrix_B))


def test_translation(so3_vector_A, translation_matrix_A):
    assert jp.all(SE3.translation(so3_vector_A) == translation_matrix_A)


def test_xyz_rpy_to_SE3(SE3_matrix_from_radians, so3_vector_A, radians_vector):
    assert jp.allclose(
        SE3_matrix_from_radians,
        SE3.xyz_rpy_to_SE3(so3_vector_A, radians_vector),
    )


def test_exp_position(exp_position_input, E3_vector_B):
    inputs = exp_position_input
    assert jp.all(SE3.exp_position(*inputs) == jp.reshape(E3_vector_B, (3, 1)))


def test_sample_SE3(sample_key):
    min_value, max_value = -0.5, 0.5
    SE3_matrix = SE3.sample(sample_key, min_value, max_value)
    rotation_matrix, position_vector = SE3.split(SE3_matrix)
    assert SE3_matrix.shape == (4, 4)
    assert jp.allclose(
        jp.dot(rotation_matrix, rotation_matrix.T), jp.eye(3), atol=1e-6
    )
    assert jp.array_equal(SE3_matrix[3, :], jp.array([0, 0, 0, 1]))
