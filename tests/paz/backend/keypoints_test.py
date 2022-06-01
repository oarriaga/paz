import numpy as np
import pytest
from paz.backend.keypoints import rotate_point2D
from paz.backend.keypoints import transform_keypoint
from paz.backend.keypoints import add_offset_to_point
from paz.backend.keypoints import rotate_keypoints3D


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


@pytest.fixture
def keypoint3D():
    keypoint = np.array([[4, 3, 9]])
    return keypoint


@pytest.fixture
def rotation_matrix():
    rotation_matrix = np.array([[0.99394977, -0.02341585, -0.10731083],
                                [0.02910355, 0.9982362, 0.05174612],
                                [0.10590983, -0.05455617, 0.99287811]])
    return rotation_matrix


@pytest.mark.parametrize("rotated_keypoint", [np.array([1, -2])])
def test_rotate_point2D(point2D_a, rotated_keypoint):
    point = rotate_point2D(point2D_a, -90)
    point = (np.array(point)).astype(np.int8)
    assert np.allclose(point, rotated_keypoint)


@pytest.mark.parametrize("transformed_keypoint", [[399.73583984,
                                                   698.07958984]])
def test_transform_keypoints(keypoint, transform_matrix, transformed_keypoint):
    point = transform_keypoint(keypoint, transform_matrix)
    assert np.allclose(point, transformed_keypoint)


@pytest.mark.parametrize("shifted_keypoint", [[150.0, 262.0]])
def test_add_offset_to_point(keypoint, offset, shifted_keypoint):
    point = add_offset_to_point(keypoint, offset)
    assert np.allclose(point, shifted_keypoint)


@pytest.mark.parametrize(
    "rotated_keypoint", [[2.93975406, 3.57683788, 9.1958738]])
def test_rotate_keypoints(rotation_matrix, keypoint3D, rotated_keypoint):
    calculated_rotated_keypoint = rotate_keypoints3D(
        np.expand_dims(rotation_matrix, 0), keypoint3D)
    assert np.allclose(rotated_keypoint, calculated_rotated_keypoint)