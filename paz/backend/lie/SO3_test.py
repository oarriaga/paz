import math

import pytest
import jax.numpy as jp
import jax

from paz import SO3


@pytest.fixture
def SO3_matrix_A():
    return jp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


@pytest.fixture
def so3_matrix_A():
    return jp.array(
        [
            [0, -1.20919958, 1.20919958],
            [1.20919958, 0, -1.20919958],
            [-1.20919958, 1.20919958, 0],
        ]
    )


@pytest.fixture
def so3_matrix_B():
    return jp.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])


@pytest.fixture
def so3_vector_B():
    return jp.array([1, 2, 3])


@pytest.fixture
def SO3_matrix_B():
    return jp.array(
        [
            [-0.69492056, 0.71352099, 0.08929286],
            [-0.19200697, -0.30378504, 0.93319235],
            [0.69297817, 0.6313497, 0.34810748],
        ]
    )


@pytest.fixture
def so3_vector_C():
    return jp.array([0, 0.866, 0.5])


@pytest.fixture
def SO3_matrix_C():
    return jp.array(
        [[0.866, -0.250, 0.433], [0.250, 0.967, 0.058], [-0.433, 0.058, 0.899]]
    )


@pytest.fixture
def radians_vector():
    """in the order of roll-pitch-yaw"""
    return jp.array([math.pi / 4, math.pi / 6, math.pi / 3])


@pytest.fixture
def sample_key():
    seed = 123
    return jax.random.PRNGKey(seed)


def test_hat(so3_vector_B, so3_matrix_B):
    assert jp.all(SO3.hat(so3_vector_B) == so3_matrix_B)


def test_vee(so3_matrix_B, so3_vector_B):
    assert jp.all(SO3.vee(so3_matrix_B) == so3_vector_B)


def test_exp(SO3_matrix_B, so3_matrix_B):
    assert jp.allclose(SO3_matrix_B, SO3.exp(so3_matrix_B))


def test_log(SO3_matrix_A, so3_matrix_A):
    assert jp.allclose(so3_matrix_A, SO3.log(SO3_matrix_A))


def test_identity_log():
    so3_matrix = SO3.log(jp.eye(3))
    jp.allclose(so3_matrix, jp.zeros((3, 3)))


def test_rodrigues(so3_vector_C, SO3_matrix_C):
    angle = math.pi / 6
    so3_matrix = SO3.compute_rodriguez_formula(angle, SO3.hat(so3_vector_C))
    jp.allclose(so3_matrix, SO3_matrix_C)


def test_rpy_to_SO3(radians_vector):
    rotation_matrix = jp.dot(
        jp.dot(
            SO3.rotation_z(radians_vector[2]), SO3.rotation_y(radians_vector[1])
        ),
        SO3.rotation_x(radians_vector[0]),
    )
    jp.allclose(rotation_matrix, SO3.rpy_to_SO3(radians_vector))


def test_compute_rotation_angle(so3_vector_B):
    desired_angle = 3.7416573867739413
    jp.allclose(desired_angle, SO3.compute_rotation_angle(so3_vector_B))


def test_sample_function(sample_key):
    SO3_matrix = SO3.sample(sample_key)
    # Assert that the matrix R is orthogonal
    assert jp.allclose(jp.dot(SO3_matrix, SO3_matrix.T), jp.eye(3), atol=1e-6)
    # Assert that the determinant is 1
    assert jp.isclose(jp.linalg.det(SO3_matrix), 1.0)
