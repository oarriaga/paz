import pytest
import numpy as np

from .backend import build_cube_points3D


@pytest.fixture
def unit_cube():
    return np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]
                     [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])


def test_build_cube_points3D(unit_cube):
    cube_points = build_cube_points3D(1, 1, 1)
    print(cube_points.shape)
    print(cube_points)
    assert np.allclose(unit_cube, cube_points)
