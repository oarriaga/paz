import numpy as np
import pytest
from paz.backend import keypoints


@pytest.fixture(params=[[2, 1]])
def point2D_a(request):
    return request.param


@pytest.fixture(params=[[1.8, 2.5]])
def point2D_b(request):
    return request.param


@pytest.fixture(params=[[149.75, 261.75]])
def keypoint(request):
    return request.param


@pytest.fixture
def transform_matrix():
    return np.array([[2.66601562e+00, 0.00000000e+00, 5.00000000e-01],
                     [2.22044605e-16, 2.66601562e+00, 2.50000000e-01]])


@pytest.fixture(params=[0.25])
def offset(request):
    return request.param


@pytest.mark.parametrize("rotated_keypoint", [np.array([1, -2])])
def test_rotate_point(point2D_a, rotated_keypoint):
    point = keypoints.rotate_keypoint(point2D_a, -90)
    point = (np.array(point)).astype(np.int8)
    assert np.allclose(point, rotated_keypoint)


@pytest.mark.parametrize("transformed_keypoint", [[399.73583984,
                                                   698.07958984]])
def test_transform_keypoints(keypoint, transform_matrix, transformed_keypoint):
    point = keypoints.transform_keypoint(keypoint, transform_matrix)
    assert np.allclose(point, transformed_keypoint)


@pytest.mark.parametrize("shifted_keypoint", [[150.0, 262.0]])
def test_add_offset_to_point(keypoint, offset, shifted_keypoint):
    point = keypoints.add_offset_to_point(keypoint, offset)
    assert np.allclose(point, shifted_keypoint)
