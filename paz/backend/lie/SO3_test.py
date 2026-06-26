import math

import pytest
import jax.numpy as jp
import jax
import jax.test_util
import jax.scipy.linalg

from paz import SO3


# Angles that stress the small-angle Taylor branch, the threshold, the
# half-turn, and the large-angle regime of the Rodrigues exponential.
EXP_EDGE_ANGLES = [
    0.0, 1e-30, 1e-12, 1e-8, 1e-4, 1e-3, 0.5,
    math.pi - 1e-3, math.pi, math.pi + 1e-2, 2 * math.pi, 10.0,
]


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
    omega = (math.pi / 6) * so3_vector_C
    rotation = SO3.compute_rodriguez_formula(SO3.hat(omega))
    assert jp.allclose(rotation, SO3_matrix_C, atol=1e-3)


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


@pytest.mark.parametrize("theta", [0.0, 1e-6, 1e-3, 1.0])
def test_exp_gradient(theta):
    f = lambda v: SO3.exp(SO3.hat(v))
    jax.test_util.check_grads(f, (jp.array([theta, 0.0, 0.0]),), order=1)


def test_exp_gradient_nonzero_at_identity():
    f = lambda v: SO3.exp(SO3.hat(v))
    jacobian = jax.jacobian(f)(jp.zeros(3))
    assert jp.allclose(jacobian[:, :, 0], SO3.hat(jp.array([1.0, 0.0, 0.0])))
    assert jp.allclose(jacobian[:, :, 1], SO3.hat(jp.array([0.0, 1.0, 0.0])))
    assert jp.allclose(jacobian[:, :, 2], SO3.hat(jp.array([0.0, 0.0, 1.0])))


@pytest.mark.parametrize("angle", [1e-6, 1e-3, 0.5, 1.0, math.pi / 2, 2.0, 3.0])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_exp_matches_matrix_exponential(angle, axis):
    omega = jp.zeros(3).at[axis].set(angle)
    expected = jax.scipy.linalg.expm(SO3.hat(omega))
    assert jp.allclose(SO3.exp(SO3.hat(omega)), expected, atol=1e-5)


@pytest.mark.parametrize("angle", EXP_EDGE_ANGLES)
def test_exp_is_orthonormal(angle):
    rotation = SO3.exp(SO3.hat(jp.array([angle, 0.0, 0.0])))
    assert jp.all(jp.isfinite(rotation))
    assert jp.allclose(rotation @ rotation.T, jp.eye(3), atol=1e-4)
    assert jp.isclose(jp.linalg.det(rotation), 1.0, atol=1e-4)


@pytest.mark.parametrize("angle", EXP_EDGE_ANGLES)
def test_exp_gradient_is_finite(angle):
    f = lambda v: SO3.exp(SO3.hat(v))
    omega = jp.array([angle, 0.2, -0.1])
    jacobian = jax.jacobian(f)(omega)
    gradient = jax.grad(lambda v: jp.sum(f(v) ** 2))(omega)
    assert jp.all(jp.isfinite(jacobian))
    assert jp.all(jp.isfinite(gradient))


@pytest.mark.parametrize("angle", [0.0, 1e-8, 1e-3, math.pi, 10.0])
def test_exp_hessian_is_finite(angle):
    loss = lambda v: jp.sum(SO3.exp(SO3.hat(v)) ** 2)
    assert jp.all(jp.isfinite(jax.hessian(loss)(jp.array([angle, 0.2, -0.1]))))


def test_exp_batch_gradient_is_finite():
    f = lambda v: SO3.exp(SO3.hat(v))
    batch = jax.random.normal(jax.random.PRNGKey(0), (64, 3)) * 3.0
    batch = batch.at[0].set(jp.zeros(3))
    assert jp.all(jp.isfinite(jax.vmap(f)(batch)))
    loss = lambda b: jp.sum(jax.vmap(f)(b) ** 2)
    assert jp.all(jp.isfinite(jax.grad(loss)(batch)))


def test_sample_function(sample_key):
    SO3_matrix = SO3.sample(sample_key)
    # Assert that the matrix R is orthogonal
    assert jp.allclose(jp.dot(SO3_matrix, SO3_matrix.T), jp.eye(3), atol=1e-6)
    # Assert that the determinant is 1
    assert jp.isclose(jp.linalg.det(SO3_matrix), 1.0)
