import pytest
import numpy as np
from paz.backend.image.draw import points3D_to_RGB


@pytest.fixture
def points3D():
    return np.array([[10, 301, 30],
                     [145, 253, 12],
                     [203, 5, 299],
                     [214, 244, 98],
                     [23, 67, 16],
                     [178, 48, 234],
                     [267, 310, 2]])


@pytest.fixture
def object_sizes():
    object_sizes = np.array([280, 260, 240])
    return object_sizes


@pytest.fixture
def object_colors():
    return np.array([[136, 166, 159],
                     [3, 119, 140],
                     [56, 132, 189],
                     [66, 110, 231],
                     [148, 193, 144],
                     [33, 174, 120],
                     [114, 175, 129]])


def test_points3D_to_RGB(points3D, object_sizes, object_colors):
    values = points3D_to_RGB(points3D, object_sizes)
    assert np.allclose(values, object_colors)
