import math

import pytest
import jax
import jax.numpy as jp

from paz import SO3
from paz import quaternion


ROTATION_VECTORS = [
    [0.0, 0.0, 0.0],
    [1e-6, 0.0, 0.0],
    [0.0, 0.0, math.pi / 2],
    [0.3, -0.2, 0.5],
    [1.0, 2.0, -1.5],
]


@pytest.mark.parametrize("rotation_vector", ROTATION_VECTORS)
def test_from_rotation_vector_matches_SO3_exp(rotation_vector):
    rotation_vector = jp.array(rotation_vector)
    value = quaternion.from_rotation_vector(rotation_vector)
    target = SO3.exp(SO3.hat(rotation_vector))
    assert jp.allclose(quaternion.to_matrix(value), target, atol=1e-6)


@pytest.mark.parametrize("rotation_vector", ROTATION_VECTORS)
def test_from_rotation_vector_has_unit_norm(rotation_vector):
    value = quaternion.from_rotation_vector(jp.array(rotation_vector))
    assert jp.allclose(jp.linalg.norm(value), 1.0, atol=1e-6)


def test_from_rotation_vector_puts_real_part_first():
    value = quaternion.from_rotation_vector(jp.array([0.0, 0.0, math.pi]))
    assert jp.allclose(value, jp.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6)


def test_from_rotation_vector_at_zero_is_identity():
    value = quaternion.from_rotation_vector(jp.zeros(3))
    assert jp.array_equal(value, jp.array([1.0, 0.0, 0.0, 0.0]))


def test_get_real_and_get_imaginary():
    value = jp.array([1.0, 2.0, 3.0, 4.0])
    assert quaternion.get_real(value) == 1.0
    assert jp.array_equal(quaternion.get_imaginary(value), value[1:])


def test_wxyz_to_xyzw():
    value = quaternion.wxyz_to_xyzw(jp.array([1.0, 2.0, 3.0, 4.0]))
    assert jp.array_equal(value, jp.array([2.0, 3.0, 4.0, 1.0]))


def test_xyzw_to_wxyz_inverts_wxyz_to_xyzw():
    value = jp.array([1.0, 2.0, 3.0, 4.0])
    round_trip = quaternion.xyzw_to_wxyz(quaternion.wxyz_to_xyzw(value))
    assert jp.array_equal(round_trip, value)


def test_compute_rates_at_zero_matches_closed_form():
    rate = jp.array([0.4, -0.3, 0.2])
    acceleration = jp.array([0.1, 0.5, -0.7])
    args = (jp.zeros(3), rate, acceleration)
    value, value_rate, value_acceleration = quaternion.compute_rates(*args)
    target_rate = jp.concatenate([jp.zeros(1), 0.5 * rate])
    real_acceleration = -0.25 * jp.dot(rate, rate)
    target_acceleration = jp.r_[real_acceleration, 0.5 * acceleration]
    assert jp.array_equal(value, jp.array([1.0, 0.0, 0.0, 0.0]))
    assert jp.allclose(value_rate, target_rate, atol=1e-6)
    assert jp.allclose(value_acceleration, target_acceleration, atol=1e-6)


@pytest.mark.parametrize("rotation_vector", ROTATION_VECTORS[2:])
def test_compute_rates_matches_finite_differences(rotation_vector):
    rate = jp.array([0.4, -0.3, 0.2])
    acceleration = jp.array([0.1, 0.5, -0.7])
    args = (jp.array(rotation_vector), rate, acceleration)
    _, value_rate, value_acceleration = quaternion.compute_rates(*args)
    curve = build_curve(*args)
    step = 1e-2
    target_rate = (curve(step) - curve(-step)) / (2 * step)
    second_difference = curve(step) - 2 * curve(0.0) + curve(-step)
    target_acceleration = second_difference / step**2
    assert jp.allclose(value_rate, target_rate, atol=1e-3)
    assert jp.allclose(value_acceleration, target_acceleration, atol=1e-2)


def build_curve(rotation_vector, rate, acceleration):
    def curve(time):
        position = rotation_vector + rate * time
        position = position + 0.5 * acceleration * time**2
        return quaternion.from_rotation_vector(position)

    return curve


def test_from_rotation_vector_gradients_at_zero_are_finite():
    def compute_real_part(rotation_vector):
        return quaternion.from_rotation_vector(rotation_vector)[0]

    hessian = jax.hessian(compute_real_part)(jp.zeros(3))
    assert jp.allclose(hessian, -0.25 * jp.eye(3), atol=1e-6)


def build_quaternion(rotation_vector):
    return quaternion.from_rotation_vector(jp.array(rotation_vector))


def test_multiply_composes_rotation_matrices():
    quaternion_A = build_quaternion([0.3, -0.2, 0.5])
    quaternion_B = build_quaternion([1.0, 2.0, -1.5])
    product = quaternion.multiply(quaternion_A, quaternion_B)
    matrix_A = quaternion.to_matrix(quaternion_A)
    matrix_B = quaternion.to_matrix(quaternion_B)
    target = matrix_A @ matrix_B
    assert jp.allclose(quaternion.to_matrix(product), target, atol=1e-6)


def test_multiply_by_conjugate_is_identity():
    value = build_quaternion([1.0, 2.0, -1.5])
    product = quaternion.multiply(value, quaternion.conjugate(value))
    assert jp.allclose(product, jp.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)


@pytest.mark.parametrize("rotation_vector", ROTATION_VECTORS)
def test_compute_angular_rate_matches_rotation_matrix(rotation_vector):
    rate = jp.array([0.4, -0.3, 0.2])
    args = (jp.array(rotation_vector),), (rate,)
    value, value_rate = jax.jvp(quaternion.from_rotation_vector, *args)
    angular_rate = quaternion.compute_angular_rate(value, value_rate)
    target = compute_body_angular_rate(*args)
    assert jp.allclose(angular_rate, target, atol=1e-5)


@pytest.mark.parametrize("rotation_vector", ROTATION_VECTORS)
def test_compute_angular_rate_of_acceleration(rotation_vector):
    rate = jp.array([0.4, -0.3, 0.2])
    acceleration = jp.array([0.1, 0.5, -0.7])
    args = (jp.array(rotation_vector), rate, acceleration)
    value, _, value_acceleration = quaternion.compute_rates(*args)
    result = quaternion.compute_angular_rate(value, value_acceleration)
    args = (jp.array(rotation_vector), rate), (rate, acceleration)
    _, target = jax.jvp(compute_body_angular_rate_at, *args)
    assert jp.allclose(result, target, atol=1e-5)


def compute_body_angular_rate_at(rotation_vector, rate):
    return compute_body_angular_rate((rotation_vector,), (rate,))


def compute_body_angular_rate(primals, tangents):
    def build_matrix(rotation_vector):
        value = quaternion.from_rotation_vector(rotation_vector)
        return quaternion.to_matrix(value)

    matrix, matrix_rate = jax.jvp(build_matrix, primals, tangents)
    return SO3.vee(matrix.T @ matrix_rate)


@pytest.mark.parametrize("rotation_vector", ROTATION_VECTORS)
def test_to_rotation_vector_inverts_from_rotation_vector(rotation_vector):
    value = quaternion.to_rotation_vector(build_quaternion(rotation_vector))
    assert jp.allclose(value, jp.array(rotation_vector), atol=1e-6)


def test_to_rotation_vector_is_the_same_for_negated_quaternion():
    value = build_quaternion([1.0, 2.0, -1.5])
    rotation_vector = quaternion.to_rotation_vector(value)
    negated = quaternion.to_rotation_vector(-value)
    assert jp.allclose(rotation_vector, negated, atol=1e-6)


def test_to_rotation_vector_at_half_turn():
    value = quaternion.to_rotation_vector(jp.array([0.0, 0.0, 0.0, 1.0]))
    assert jp.allclose(value, jp.array([0.0, 0.0, math.pi]), atol=1e-6)


def test_to_rotation_vector_ignores_scale():
    value = build_quaternion([0.3, -0.2, 0.5])
    rotation_vector = quaternion.to_rotation_vector(value)
    scaled = quaternion.to_rotation_vector(2.0 * value)
    assert jp.allclose(rotation_vector, scaled, atol=1e-6)


def test_to_rotation_vector_jacobian_at_identity():
    identity = jp.array([1.0, 0.0, 0.0, 0.0])
    jacobian = jax.jacfwd(quaternion.to_rotation_vector)(identity)
    target = jp.concatenate([jp.zeros((3, 1)), 2.0 * jp.eye(3)], axis=1)
    assert jp.allclose(jacobian, target, atol=1e-6)


def test_to_rotation_vector_jacobian_is_continuous_at_series_switch():
    below = jp.array([1.0, 0.99e-4, 0.0, 0.0])
    above = jp.array([1.0, 1.01e-4, 0.0, 0.0])
    compute_jacobian = jax.jacfwd(quaternion.to_rotation_vector)
    jacobian_below = compute_jacobian(below)
    jacobian_above = compute_jacobian(above)
    assert jp.allclose(jacobian_below, jacobian_above, atol=1e-4)
