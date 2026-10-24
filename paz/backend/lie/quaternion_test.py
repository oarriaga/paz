import jax
import jax.numpy as jp
import pytest

from paz import SO3
from paz import quaternion


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_from_matrix_round_trips_sampled_rotation(seed):
    rotation = SO3.sample(jax.random.PRNGKey(seed))
    recovered = quaternion.to_matrix(quaternion.from_matrix(rotation))
    assert jp.allclose(recovered, rotation, atol=1e-6)


def test_from_matrix_identity():
    expected = jp.array([1.0, 0.0, 0.0, 0.0])
    assert jp.allclose(quaternion.from_matrix(jp.eye(3)), expected)


def test_from_matrix_half_turn_keeps_unit_norm():
    rotation = SO3.rotation_x(jp.pi)
    unit = quaternion.from_matrix(rotation)
    assert jp.allclose(jp.linalg.norm(unit), 1.0, atol=1e-6)
    assert jp.allclose(quaternion.to_matrix(unit), rotation, atol=1e-6)


def test_slerp_endpoints_and_midpoint():
    quaternion_0 = quaternion.from_matrix(jp.eye(3))
    quaternion_1 = quaternion.from_matrix(SO3.rotation_z(jp.pi / 2))
    assert jp.allclose(quaternion.slerp(quaternion_0, quaternion_1, 0.0),
                       quaternion_0, atol=1e-6)
    assert jp.allclose(quaternion.slerp(quaternion_0, quaternion_1, 1.0),
                       quaternion_1, atol=1e-6)
    midpoint = quaternion.slerp(quaternion_0, quaternion_1, 0.5)
    expected = quaternion.to_matrix(midpoint)
    assert jp.allclose(expected, SO3.rotation_z(jp.pi / 4), atol=1e-6)


def test_slerp_identical_quaternions_are_preserved():
    unit = quaternion.from_matrix(SO3.sample(jax.random.PRNGKey(5)))
    assert jp.allclose(quaternion.slerp(unit, unit, 0.3), unit, atol=1e-6)


def test_slerp_takes_the_short_arc():
    unit = quaternion.from_matrix(SO3.rotation_z(0.2))
    assert jp.allclose(quaternion.slerp(unit, -unit, 0.5), unit, atol=1e-6)
