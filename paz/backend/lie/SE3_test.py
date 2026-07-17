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


def build_tangent(angle):
    axis = jp.array([0.36, 0.48, 0.8])
    linear = jp.array([0.5, -0.3, 0.2])
    return jp.concatenate([angle * axis, linear])


@pytest.fixture
def pose_A():
    return SE3.sample(jax.random.PRNGKey(1), -1.0, 1.0)


@pytest.fixture
def pose_B():
    return SE3.sample(jax.random.PRNGKey(2), -1.0, 1.0)


def test_between_recovers_composed(pose_A, pose_B):
    composed = SE3.compose(pose_A, pose_B)
    assert jp.allclose(SE3.between(pose_A, composed), pose_B, atol=1e-5)


@pytest.mark.parametrize("seed", range(6))
def test_retract_local_coordinates_round_trip(seed):
    key_A, key_B = jax.random.split(jax.random.PRNGKey(seed))
    pose_A = SE3.sample(key_A, -1.0, 1.0)
    pose_B = SE3.sample(key_B, -1.0, 1.0)
    delta = SE3.local_coordinates(pose_A, pose_B)
    assert jp.allclose(SE3.retract(pose_A, delta), pose_B, atol=2e-3)


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.5, 1.0, 2.0])
def test_Log_Exp_round_trip(angle):
    tangent = build_tangent(angle)
    assert jp.allclose(SE3.Log(SE3.Exp(tangent)), tangent, atol=1e-5)


def test_Log_Exp_round_trip_tiny_tangent():
    tangent = jp.full(6, 1e-8)
    assert jp.allclose(SE3.Log(SE3.Exp(tangent)), tangent, atol=1e-7)


@pytest.mark.parametrize("angle", [math.pi - 0.3, math.pi - 0.1])
def test_Log_Exp_round_trip_near_pi(angle):
    tangent = build_tangent(angle)
    assert jp.allclose(SE3.Log(SE3.Exp(tangent)), tangent, atol=1e-3)


@pytest.mark.parametrize("seed", range(6))
def test_Exp_Log_round_trip(seed):
    pose = SE3.sample(jax.random.PRNGKey(seed), -1.0, 1.0)
    assert jp.allclose(SE3.Exp(SE3.Log(pose)), pose, atol=1e-3)


def test_retract_at_identity():
    tangent = build_tangent(0.7)
    retracted = SE3.retract(jp.eye(4), tangent)
    assert jp.allclose(retracted, SE3.Exp(tangent), atol=1e-6)


def test_retract_first_order(pose_A):
    small_tangent = 1e-3 * build_tangent(0.7)
    linearized = pose_A + pose_A @ SE3.hat(small_tangent)
    retracted = SE3.retract(pose_A, small_tangent)
    assert jp.allclose(retracted, linearized, atol=1e-5)


def test_left_jacobian_at_zero():
    assert jp.allclose(SE3.left_jacobian(jp.zeros(6)), jp.eye(6))


JACOBIAN_ANGLES = [1e-8, 0.5, 1.5, 2.5, math.pi - 1e-3, math.pi]


@pytest.mark.parametrize("angle", JACOBIAN_ANGLES)
def test_left_jacobian_inverse_consistency(angle):
    tangent = build_tangent(angle)
    jacobian = SE3.left_jacobian(tangent)
    inverse = SE3.left_jacobian_inverse(tangent)
    assert jp.allclose(jacobian @ inverse, jp.eye(6), atol=1e-5)


def test_left_jacobian_inverse_random_tangents():
    tangents = jax.random.normal(jax.random.PRNGKey(9), (32, 6))
    jacobians = jax.vmap(SE3.left_jacobian)(tangents)
    inverses = jax.vmap(SE3.left_jacobian_inverse)(tangents)
    products = jp.einsum("bij,bjk->bik", jacobians, inverses)
    assert jp.allclose(products, jp.eye(6), atol=1e-5)


@pytest.mark.parametrize("angle", [0.0, 0.5, 1.5, math.pi - 0.1])
def test_left_jacobian_first_order(angle):
    tangent = build_tangent(angle)
    delta = jax.random.normal(jax.random.PRNGKey(4), (6,))
    delta = 1e-2 * delta / jp.linalg.norm(delta)
    perturbed = SE3.Exp(tangent + delta)
    step = SE3.Exp(SE3.left_jacobian(tangent) @ delta)
    approximated = SE3.compose(step, SE3.Exp(tangent))
    residual = SE3.local_coordinates(approximated, perturbed)
    assert jp.linalg.norm(residual) < 1e-4


def compute_difference_column(tangent, direction, step):
    inverse = SE3.invert(SE3.Exp(tangent))
    plus = SE3.Log(SE3.Exp(tangent + step * direction) @ inverse)
    minus = SE3.Log(SE3.Exp(tangent - step * direction) @ inverse)
    return (plus - minus) / (2.0 * step)


@pytest.mark.parametrize("angle", [0.0, 0.5, 1.5, math.pi - 0.1])
def test_left_jacobian_matches_finite_differences(angle):
    tangent = build_tangent(angle)
    columns = []
    for index in range(6):
        direction = jp.zeros(6).at[index].set(1.0)
        columns.append(compute_difference_column(tangent, direction, 1e-2))
    difference_jacobian = jp.stack(columns, axis=1)
    assert jp.allclose(difference_jacobian, SE3.left_jacobian(tangent),
                       atol=2e-4)


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.5, math.pi - 1e-3, math.pi])
def test_retract_gradient_is_finite(pose_A, angle):
    loss = lambda tangent: jp.sum(SE3.retract(pose_A, tangent) ** 2)
    assert jp.all(jp.isfinite(jax.grad(loss)(build_tangent(angle))))


def test_retract_gradient_is_finite_at_zero_tangent(pose_A):
    loss = lambda tangent: jp.sum(SE3.retract(pose_A, tangent) ** 2)
    assert jp.all(jp.isfinite(jax.grad(loss)(jp.zeros(6))))


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.5, math.pi - 0.1])
def test_local_coordinates_gradient_is_finite(pose_A, angle):
    def loss(tangent):
        moved = SE3.retract(pose_A, tangent)
        return jp.sum(SE3.local_coordinates(pose_A, moved) ** 2)

    assert jp.all(jp.isfinite(jax.grad(loss)(build_tangent(angle))))


def test_local_coordinates_gradient_is_finite_at_zero_tangent(pose_A):
    def loss(tangent):
        moved = SE3.retract(pose_A, tangent)
        return jp.sum(SE3.local_coordinates(pose_A, moved) ** 2)

    assert jp.all(jp.isfinite(jax.grad(loss)(jp.zeros(6))))


def test_jit_and_vmap_over_batch():
    keys = jax.random.split(jax.random.PRNGKey(8), 64)
    sample_pose = lambda key: SE3.sample(key, -1.0, 1.0)
    poses_A = jax.vmap(sample_pose)(keys[:32])
    poses_B = jax.vmap(sample_pose)(keys[32:])
    tangents = jax.random.normal(jax.random.PRNGKey(9), (32, 6))
    retracted = jax.jit(jax.vmap(SE3.retract))(poses_A, tangents)
    deltas = jax.jit(jax.vmap(SE3.local_coordinates))(poses_A, poses_B)
    jacobians = jax.jit(jax.vmap(SE3.left_jacobian))(tangents)
    inverses = jax.jit(jax.vmap(SE3.left_jacobian_inverse))(tangents)
    assert jp.all(jp.isfinite(retracted))
    assert jp.all(jp.isfinite(deltas))
    assert jp.all(jp.isfinite(jacobians))
    assert jp.all(jp.isfinite(inverses))
    recovered = jax.vmap(SE3.retract)(poses_A, deltas)
    assert jp.allclose(recovered, poses_B, atol=1e-3)
