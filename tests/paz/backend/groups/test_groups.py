import pytest
import numpy as np

from paz.backend.groups import homogenous_quaternion_to_rotation_matrix
from paz.backend.groups import quaternion_to_rotation_matrix
from paz.backend.groups import rotation_vector_to_rotation_matrix
from paz.backend.groups import to_affine_matrix
from paz.backend.groups import build_rotation_matrix_x
from paz.backend.groups import build_rotation_matrix_y
from paz.backend.groups import build_rotation_matrix_z
from paz.backend.groups import compute_norm_SO3
from paz.backend.groups import calculate_canonical_rotation
from paz.backend.groups import rotation_matrix_to_axis_angle
from paz.backend.groups import rotation_matrix_to_compact_axis_angle


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


def test_calculate_canonical_rotation(rotation_matrix_X_HALF_PI):
    X_PI = np.matmul(rotation_matrix_X_HALF_PI, rotation_matrix_X_HALF_PI)
    rotations = [X_PI, rotation_matrix_X_HALF_PI]
    canonical_rotation = calculate_canonical_rotation(np.eye(3), rotations)
    assert np.allclose(
        canonical_rotation, np.linalg.inv(rotation_matrix_X_HALF_PI))


@pytest.fixture
def rotation_matrix():
    rotation_matrix = np.array([[0.99394977, -0.02341585, -0.10731083],
                                [0.02910355, 0.9982362, 0.05174612],
                                [0.10590983, -0.05455617, 0.99287811]])
    return rotation_matrix


@pytest.mark.parametrize(
    "axis_angle", [[-0.43571813, -0.87396149, 0.21526963, 0.12228879]])
def test_rotation_matrix_to_axis_angle(rotation_matrix, axis_angle):
    estimated_axis_angle = rotation_matrix_to_axis_angle(rotation_matrix)
    assert np.allclose(axis_angle, estimated_axis_angle)


@pytest.mark.parametrize(
    "compact_axis_angle", [[-0.05328344, -0.10687569, 0.02632506]])
def test_rotation_matrix_to_compact_axis_angle(
        rotation_matrix, compact_axis_angle):
    estimated_compact_axis_angle = rotation_matrix_to_compact_axis_angle(
        rotation_matrix)
    assert np.allclose(compact_axis_angle, estimated_compact_axis_angle)