import pytest
import numpy as np
from paz.backend.groups.quaternion import rotation_vector_to_quaternion
from paz.backend.groups.quaternion import get_quaternion_conjugate
from paz.backend.groups.quaternion import rotation_matrix_to_quaternion
from paz.backend.groups.quaternion import quaternion_to_rotation_matrix


@pytest.fixture
def quaternion_target():
    return np.array([0.45936268, -0.45684629, 0.04801626, 0.76024458])


@pytest.fixture
def rotation_vector():
    return np.array([1., -0.994522, 0.104528])


@pytest.fixture
def rotation_matrix():
    rotation_matrix = np.array([[0.99394977, -0.02341585, -0.10731083],
                                [0.02910355, 0.9982362, 0.05174612],
                                [0.10590983, -0.05455617, 0.99287811]])
    return rotation_matrix


@pytest.fixture()
def quaternion():
    return np.array([-0.02662533, -0.05340496, 0.01315443, 0.99813126])


def test_rotation_vector_to_quaternion(rotation_vector, quaternion_target):
    result = rotation_vector_to_quaternion(rotation_vector)
    assert np.allclose(result, quaternion_target)


@pytest.mark.parametrize("quaternion_conjugate",
                         [[0.02662533, 0.05340496, -0.01315443, 0.99813126]])
def test_get_quaternion_conjugate(quaternion, quaternion_conjugate):
    estimated_quaternion_conjugate = get_quaternion_conjugate(quaternion)
    assert np.allclose(quaternion_conjugate, estimated_quaternion_conjugate)


def test_rotation_matrix_to_quaternion(rotation_matrix, quaternion):
    estimated_quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    estimates_rotation_matrix = quaternion_to_rotation_matrix(
        estimated_quaternion)
    assert np.allclose(quaternion, estimated_quaternion)
    assert np.allclose(rotation_matrix, estimates_rotation_matrix)
