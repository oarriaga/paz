import pytest
import numpy as np
from paz.backend.groups.quaternion import rotation_vector_to_quaternion


@pytest.fixture
def quaternion_target():
    return np.array([0.45936268, -0.45684629, 0.04801626, 0.76024458])


@pytest.fixture
def rotation_vector():
    return np.array([1., -0.994522, 0.104528])


def test_rotation_vector_to_quaternion(rotation_vector, quaternion_target):
    result = rotation_vector_to_quaternion(rotation_vector)
    assert np.allclose(result, quaternion_target)
