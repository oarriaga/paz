import pytest
import numpy as np

from .backend import build_cube_points3D
from .backend import preprocess_image_points2D
from .backend import replace_lower_than_threshold
from .backend import arguments_to_image_points2D
from .backend import normalize_points2D
from .backend import denormalize_points2D
from .backend import homogenous_quaternion_to_rotation_matrix
from .backend import quaternion_to_rotation_matrix
from .backend import rotation_vector_to_rotation_matrix
from .backend import to_affine_matrix
from .backend import image_to_normalized_device_coordinates
from .backend import normalized_device_coordinates_to_image
from .backend import build_rotation_matrix_x
from .backend import build_rotation_matrix_y
from .backend import build_rotation_matrix_z
from .backend import compute_norm_SO3
from .backend import calculate_canonical_rotation
from .backend import normalize_min_max
from .backend import extract_bounding_box_corners
from .backend import compute_vertices_colors
from .backend import project_to_image
from .backend import points3D_to_RGB


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
    image_points2D = preprocess_image_points2D(points2D)
    num_points = len(points2D)
    assert image_points2D.shape == (num_points, 1, 2)
    assert image_points2D.data.contiguous
    assert np.allclose(np.squeeze(image_points2D, 1), points2D)


# def test_solve_PnP_RANSAC(object_points3D, image_points2D, camera_intrinsics,
# def test_project_to_image(rotation, translation, points3D, camera_intrisincs)
# def draw_cube

def test_replace_lower_than_threshold():
    source = np.ones((128, 128, 3))
    target = replace_lower_than_threshold(source, 2.0, 5.0)
    assert np.allclose(target, 5.0)

    source = np.ones((128, 128, 3))
    target = replace_lower_than_threshold(source, 0.0, -1.0)
    assert np.allclose(target, 1.0)


def test_arguments_to_image_points2D():
    col_args = np.array([3, 44, 6])
    row_args = np.array([66, 0, 5])
    image_points2D = arguments_to_image_points2D(row_args, col_args)
    assert np.allclose(image_points2D, np.array([[3, 66], [44, 0], [6, 5]]))


# def test_points3D_to_RGB(points3D):
# def draw_mask
# def draw_masks
# def draw_points2D

def test_normalize_points2D():
    height, width = 480, 640
    points2D = np.array([[0, 0], [320, 240], [640, 480]])
    normalized_points = normalize_points2D(points2D, height, width)
    assert np.allclose(normalized_points, np.array([[-1, -1], [0, 0], [1, 1]]))


def test_denormalize_points2D():
    height, width = 480, 640
    normalized_points = np.array([[-1, -1], [0, 0], [1, 1]])
    points2D = denormalize_points2D(normalized_points, height, width)
    assert np.allclose(points2D, np.array([[0, 0], [320, 240], [640, 480]]))

# def draw_pose6D
# def draw_poses6D


def test_homogenous_quaternion_to_rotation_matrix_identity():
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    matrix = homogenous_quaternion_to_rotation_matrix(quaternion)
    assert np.allclose(np.eye(3), matrix)


def test_homogenous_quaternion_to_rotation_matrix_Z(rotation_matrix_Z_HALF_PI):
    quaternion = np.array([0, 0, 0.7071068, 0.7071068])
    matrix = homogenous_quaternion_to_rotation_matrix(quaternion)
    assert np.allclose(rotation_matrix_Z_HALF_PI, matrix)


def test_homogenous_quaternion_to_rotation_matrix_Y(rotation_matrix_Y_HALF_PI):
    quaternion = np.array([0, 0.7071068, 0.0, 0.7071068])
    matrix = homogenous_quaternion_to_rotation_matrix(quaternion)
    assert np.allclose(rotation_matrix_Y_HALF_PI, matrix)


def test_homogenous_quaternion_to_rotation_matrix_X(rotation_matrix_X_HALF_PI):
    quaternion = np.array([0.7071068, 0.0, 0.0, 0.7071068])
    matrix = homogenous_quaternion_to_rotation_matrix(quaternion)
    assert np.allclose(rotation_matrix_X_HALF_PI, matrix)


def test_quaternion_to_rotation_matrix_identity():
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    matrix = quaternion_to_rotation_matrix(quaternion)
    assert np.allclose(np.eye(3), matrix)


def test_quaternion_to_rotation_matrix_Z(rotation_matrix_Z_HALF_PI):
    quaternion = np.array([0, 0, 0.7071068, 0.7071068])
    matrix = quaternion_to_rotation_matrix(quaternion)
    assert np.allclose(rotation_matrix_Z_HALF_PI, matrix)


def test_quaternion_to_rotation_matrix_Y(rotation_matrix_Y_HALF_PI):
    quaternion = np.array([0, 0.7071068, 0.0, 0.7071068])
    matrix = quaternion_to_rotation_matrix(quaternion)
    assert np.allclose(rotation_matrix_Y_HALF_PI, matrix)


def test_quaternion_to_rotation_matrix_X(rotation_matrix_X_HALF_PI):
    quaternion = np.array([0.7071068, 0.0, 0.0, 0.7071068])
    matrix = quaternion_to_rotation_matrix(quaternion)
    assert np.allclose(rotation_matrix_X_HALF_PI, matrix)


def test_rotation_vector_to_rotation_matrix_identity():
    rotation_vector = np.array([0.0, 0.0, 0.0])
    matrix = rotation_vector_to_rotation_matrix(rotation_vector)
    assert np.allclose(np.eye(3), matrix)


def test_rotation_vector_to_rotation_matrix_Z(rotation_matrix_Z_HALF_PI):
    rotation_vector = np.array([0.0, 0.0, np.pi / 2.0])
    matrix = rotation_vector_to_rotation_matrix(rotation_vector)
    assert np.allclose(rotation_matrix_Z_HALF_PI, matrix)


def test_rotation_vector_to_rotation_matrix_Y(rotation_matrix_Y_HALF_PI):
    rotation_vector = np.array([0.0, np.pi / 2.0, 0.0])
    matrix = rotation_vector_to_rotation_matrix(rotation_vector)
    assert np.allclose(rotation_matrix_Y_HALF_PI, matrix)


def test_rotation_vector_to_rotation_matrix_X(rotation_matrix_X_HALF_PI):
    rotation_vector = np.array([np.pi / 2.0, 0.0, 0.0])
    matrix = rotation_vector_to_rotation_matrix(rotation_vector)
    assert np.allclose(rotation_matrix_X_HALF_PI, matrix)


def test_to_affine_matrix_identity():
    rotation_matrix = np.eye(3)
    translation = np.zeros(3)
    matrix = to_affine_matrix(rotation_matrix, translation)
    assert np.allclose(matrix, np.eye(4))


def test_to_affine_matrix():
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0],
                                [0.0, 1.0, 0.0]])
    translation = np.array([3.0, 1.2, 3.0])
    matrix = to_affine_matrix(rotation_matrix, translation)
    affine_matrix = np.array([[1.0, 0.0, 0.0, 3.0],
                              [0.0, 0.0, -1.0, 1.2],
                              [0.0, 1.0, 0.0, 3.0],
                              [0.0, 0.0, 0.0, 1.0]])
    assert np.allclose(affine_matrix, matrix)


def test_image_to_normalized_device_coordinates():
    image = np.array([[0, 127.5, 255]])
    values = image_to_normalized_device_coordinates(image)
    assert np.allclose(values, np.array([[-1.0, 0.0, 1.0]]))


def test_normalized_device_coordinates_to_image():
    coordinates = np.array([[-1.0, 0.0, 1.0]])
    values = normalized_device_coordinates_to_image(coordinates)
    assert np.allclose(values, np.array([[0.0, 127.5, 255.0]]))


def test_build_rotation_matrix_x(rotation_matrix_X_HALF_PI):
    angle = np.pi / 2.0
    matrix = build_rotation_matrix_x(angle)
    assert np.allclose(matrix, rotation_matrix_X_HALF_PI)


def test_build_rotation_matrix_y(rotation_matrix_Y_HALF_PI):
    angle = np.pi / 2.0
    matrix = build_rotation_matrix_y(angle)
    assert np.allclose(matrix, rotation_matrix_Y_HALF_PI)


def test_build_rotation_matrix_z(rotation_matrix_Z_HALF_PI):
    angle = np.pi / 2.0
    matrix = build_rotation_matrix_z(angle)
    assert np.allclose(matrix, rotation_matrix_Z_HALF_PI)


# test_sample_uniform
# test_sample_inside_box3D
# test_sample_front_rotation_matrix
# test_sample_afine_transform
# test_sample_random_rotation_matrix

def test_compute_norm_SO3_X(rotation_matrix_X_HALF_PI):
    norm = compute_norm_SO3(np.eye(3), rotation_matrix_X_HALF_PI)
    assert np.allclose(norm, 2.0)


def test_compute_norm_SO3_Y(rotation_matrix_Y_HALF_PI):
    norm = compute_norm_SO3(np.eye(3), rotation_matrix_Y_HALF_PI)
    assert np.allclose(norm, 2.0)


def test_compute_norm_SO3_Z(rotation_matrix_Z_HALF_PI):
    norm = compute_norm_SO3(np.eye(3), rotation_matrix_Z_HALF_PI)
    assert np.allclose(norm, 2.0)


def test_compute_norm_SO3_identity():
    norm = compute_norm_SO3(np.eye(3), np.eye(3))
    assert np.allclose(norm, 0.0)


def test_compute_norm_SO3_X_to_Z(rotation_matrix_X_HALF_PI,
                                 rotation_matrix_Z_HALF_PI):
    norm = compute_norm_SO3(rotation_matrix_X_HALF_PI,
                            rotation_matrix_Z_HALF_PI)
    assert np.allclose(norm, 2.449489742783178)


# calculate_canonical_rotation


def test_normalize_min_max():
    x = np.array([-1.0, 0.0, 1.0])
    values = normalize_min_max(x, np.min(x), np.max(x))
    assert np.allclose(values, np.array([0.0, 0.5, 1.0]))


def test_extract_corners3D(points3D):
    bottom_left, top_right = extract_bounding_box_corners(points3D)
    assert np.allclose(bottom_left, np.array([10, 5, 2]))
    assert np.allclose(top_right, np.array([267, 310, 299]))


def test_compute_vertices_colors(points3D):
    values = compute_vertices_colors(points3D)
    colors = np.array([[0, 247, 24],
                       [133, 207, 8],
                       [191, 0, 255],
                       [202, 199, 82],
                       [12, 51, 12],
                       [166, 35, 199],
                       [255, 255, 0]])
    assert np.allclose(values, colors)


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


def test_calculate_canonical_rotation(rotation_matrix_X_HALF_PI):
    X_PI = np.matmul(rotation_matrix_X_HALF_PI, rotation_matrix_X_HALF_PI)
    rotations = [X_PI, rotation_matrix_X_HALF_PI]
    canonical_rotation = calculate_canonical_rotation(np.eye(3), rotations)
    assert np.allclose(
        canonical_rotation, np.linalg.inv(rotation_matrix_X_HALF_PI))


def test_points3D_to_RGB(points3D, object_sizes, object_colors):
    values = points3D_to_RGB(points3D, object_sizes)
    assert np.allclose(values, object_colors)
