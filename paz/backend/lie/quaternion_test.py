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
