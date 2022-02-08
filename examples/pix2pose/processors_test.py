import pytest
import numpy as np

from .processors import ImageToNormalizedDeviceCoordinates
from .processors import NormalizedDeviceCoordinatesToImage
from .processors import ReplaceLowerThanThreshold
from .processors import NormalizePoints2D
from .processors import ToAffineMatrix
from .processors import ArgumentsToImagePoints2D
from .processors import UnwrapDictionary
# from .processors import GetNonZeroArguments
# from .processors import GetNonZeroValues
from .processors import Scale


@pytest.fixture
def rotation_matrix_X_HALF_PI():
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0],
                                [0.0, 1.0, 0.0]])
    return rotation_matrix


@pytest.fixture
def rotation_matrix_Y_HALF_PI():
    rotation_matrix = np.array([[0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0],
                                [-1.0, 0.0, 0.0]])
    return rotation_matrix


@pytest.fixture
def rotation_matrix_Z_HALF_PI():
    rotation_matrix = np.array([[0.0, -1.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]])
    return rotation_matrix


@pytest.fixture
def object_sizes():
    object_sizes = np.array([280, 260, 240])
    return object_sizes


def test_ImageToNormalizedDeviceCoordinates():
    image = np.array([[0, 127.5, 255]])
    image_to_NDC = ImageToNormalizedDeviceCoordinates()
    values = image_to_NDC(image)
    assert np.allclose(values, np.array([[-1.0, 0.0, 1.0]]))


def test_NormalizedDeviceCoordinatesToImage():
    coordinates = np.array([[-1.0, 0.0, 1.0]])
    NDC_to_image = NormalizedDeviceCoordinatesToImage()
    values = NDC_to_image(coordinates)
    assert np.allclose(values, np.array([[0.0, 127.5, 255.0]]))


def test_ReplaceLowerThanThreshold():
    source = np.ones((128, 128, 3))
    replace_lower_than_threshold = ReplaceLowerThanThreshold(2.0, 5.0)
    target = replace_lower_than_threshold(source)
    assert np.allclose(target, 5.0)

    source = np.ones((128, 128, 3))
    replace_lower_than_threshold = ReplaceLowerThanThreshold(0.0, -1.0)
    target = replace_lower_than_threshold(source)
    assert np.allclose(target, 1.0)


def test_NormalizePoints2D():
    height, width = 480, 640
    points2D = np.array([[0, 0], [320, 240], [640, 480]])
    normalize_points2D = NormalizePoints2D((height, width))
    normalized_points = normalize_points2D(points2D, height, width)
    assert np.allclose(normalized_points, np.array([[-1, -1], [0, 0], [1, 1]]))


def test_ToAffineMarixIdentity():
    rotation_matrix = np.eye(3)
    translation = np.zeros(3)
    to_affine_matrix = ToAffineMatrix()
    matrix = to_affine_matrix(rotation_matrix, translation)
    assert np.allclose(matrix, np.eye(4))


def test_ToAffineMatrix():
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0],
                                [0.0, 1.0, 0.0]])
    translation = np.array([3.0, 1.2, 3.0])
    to_affine_matrix = ToAffineMatrix()
    matrix = to_affine_matrix(rotation_matrix, translation)
    affine_matrix = np.array([[1.0, 0.0, 0.0, 3.0],
                              [0.0, 0.0, -1.0, 1.2],
                              [0.0, 1.0, 0.0, 3.0],
                              [0.0, 0.0, 0.0, 1.0]])
    assert np.allclose(affine_matrix, matrix)


def test_ArgumentsToImagePoints2D():
    col_args = np.array([3, 44, 6])
    row_args = np.array([66, 0, 5])
    arguments_to_image_points2D = ArgumentsToImagePoints2D()
    image_points2D = arguments_to_image_points2D(row_args, col_args)
    assert np.allclose(image_points2D, np.array([[3, 66], [44, 0], [6, 5]]))


def test_UnwrapDictionary():
    dictionary = {'a': 1, 'b': 2, 'c': 3}
    unwrap = UnwrapDictionary(['b', 'a', 'c'])
    assert unwrap(dictionary) == [2, 1, 3]


def test_Scale(object_sizes):
    scale = Scale(object_sizes)
    values = np.array([1.0, 0.5, 0.25])
    scaled_values = scale(values)
    assert np.allclose(scaled_values, values * object_sizes)
