import pytest
import numpy as np

from paz.backend.keypoints import build_cube_points3D
from paz.backend.keypoints import _preprocess_image_points2D
from paz.backend.keypoints import denormalize_keypoints2D
from paz.backend.keypoints import normalize_keypoints2D
from paz.backend.keypoints import arguments_to_image_points2D
from paz.backend.keypoints import project_to_image


@pytest.fixture
def unit_cube():
    return np.array([[0.5, -0.5, 0.5],
                     [0.5, -0.5, -0.5],
                     [-0.5, -0.5, -0.5],
                     [-0.5, -0.5, 0.5],
                     [0.5, 0.5, 0.5],
                     [0.5, 0.5, -0.5],
                     [-0.5, 0.5, -0.5],
                     [-0.5, 0.5, 0.5]])


@pytest.fixture
def points2D():
    return np.array([[10, 301],
                     [145, 253],
                     [203, 5],
                     [214, 244],
                     [23, 67],
                     [178, 48],
                     [267, 310]])


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
def object_colors():
    return np.array([[136, 166, 159],
                     [3, 119, 140],
                     [56, 132, 189],
                     [66, 110, 231],
                     [148, 193, 144],
                     [33, 174, 120],
                     [114, 175, 129]])


@pytest.fixture
def object_sizes():
    object_sizes = np.array([280, 260, 240])
    return object_sizes


def test_build_cube_points3D(unit_cube):
    cube_points = build_cube_points3D(1, 1, 1)
    assert np.allclose(unit_cube, cube_points)


def test_preprocess_image_point2D(points2D):
    image_points2D = _preprocess_image_points2D(points2D)
    num_points = len(points2D)
    assert image_points2D.shape == (num_points, 1, 2)
    assert image_points2D.data.contiguous
    assert np.allclose(np.squeeze(image_points2D, 1), points2D)


def test_arguments_to_image_points2D():
    col_args = np.array([3, 44, 6])
    row_args = np.array([66, 0, 5])
    image_points2D = arguments_to_image_points2D(row_args, col_args)
    assert np.allclose(image_points2D, np.array([[3, 66], [44, 0], [6, 5]]))


def test_normalize_points2D():
    height, width = 480, 640
    points2D = np.array([[0, 0], [320, 240], [640, 480]])
    normalized_points = normalize_keypoints2D(points2D, height, width)
    assert np.allclose(normalized_points, np.array([[-1, -1], [0, 0], [1, 1]]))


def test_denormalize_points2D():
    height, width = 480, 640
    normalized_points = np.array([[-1, -1], [0, 0], [1, 1]])
    points2D = denormalize_keypoints2D(normalized_points, height, width)
    assert np.allclose(points2D, np.array([[0, 0], [320, 240], [640, 480]]))


def test_project_to_image():
    points3D = np.array([[1.0, 1.0, 1.0]])
    translation = np.array([0.0, 0.0, -3.0])
    rotation = np.array([[0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0]])
    fx = 1.0
    fy = 1.0
    tx = 0.0
    ty = 0.0
    camera_intrinsics = np.array([[fx, 0.0, tx], [0.0, fy, ty]])
    points2D = project_to_image(rotation, translation,
                                points3D, camera_intrinsics)
    assert np.allclose(points2D, np.array([0.5, -0.5]))
