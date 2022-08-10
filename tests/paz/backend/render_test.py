import pytest
import numpy as np

from paz.backend.standard import calculate_norm
from paz.backend.render import sample_point_in_full_sphere
from paz.backend.render import sample_point_in_top_sphere
from paz.backend.render import sample_point_in_sphere
from paz.backend.render import random_perturbation
from paz.backend.render import random_translation
from paz.backend.render import get_look_at_transform
from paz.backend.render import compute_modelview_matrices
from paz.backend.render import scale_translation
from paz.backend.render import sample_uniformly
from paz.backend.render import split_alpha_channel
from paz.backend.render import roll_camera
from paz.backend.render import translate_camera


@pytest.fixture
def origin_A():
    return np.array([0.0, 0.0, 0.0])


@pytest.fixture
def target_A():
    return np.array([1.0, 1.0, 1.0])


@pytest.fixture
def transform_A():
    return np.array([[-0.70710678, 0.0, 0.70710678, 0.0],
                     [-0.40824829, 0.81649658, -0.40824829, 0.0],
                     [-0.57735027, -0.57735027, -0.57735027, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


@pytest.fixture
def origin_B():
    return np.array([3.4, 0.1, -5.0])


@pytest.fixture
def target_B():
    return np.array([-4.1, 2.0, -1.1])


@pytest.fixture
def transform_B():
    return np.array([[-0.46135274, 0.0, -0.8872168, -2.8674847],
                     [0.19455847, 0.97565955, -0.10117041, -1.2649168],
                     [0.86562154, -0.21929079, -0.4501232, -5.17180017],
                     [0.0, 0.0, 0.0, 1.0]])


def test_sample_point_in_full_sphere():
    for distance in range(1, 100):
        distance = distance + np.random.rand()
        norm = calculate_norm(sample_point_in_full_sphere(distance))
        assert np.isclose(norm, distance)


def test_sample_point_in_top_sphere():
    for distance in range(1, 100):
        distance = distance + np.random.rand()
        point = sample_point_in_top_sphere(distance)
        assert point[1] > 0  # tests top
        assert np.isclose(calculate_norm(point), distance)


def test_sample_point_in_sphere():
    for distance in range(1, 100):
        distance = distance + np.random.rand()
        point = sample_point_in_sphere(distance, top_only=True)
        assert point[1] > 0  # tests top
        assert np.isclose(calculate_norm(point), distance)
        norm = calculate_norm(sample_point_in_sphere(distance, top_only=False))
        assert np.isclose(norm, distance)


def test_random_perturbation():
    for trials in range(100):
        localization = np.random.rand((3)) * np.random.uniform(1, 100)
        perturbation = random_perturbation(localization, 0)
        assert np.allclose(perturbation, localization)
        shift = np.random.uniform(-3, 3)
        perturbation = random_perturbation(localization, shift)


def test_random_translation():
    for trials in range(100):
        localization = np.random.rand((3)) * np.random.uniform(1, 100)
        perturbation = random_translation(localization, 0)
        assert np.allclose(perturbation, localization)


def test_split_alpha_channel():
    image = np.random.rand(100, 100, 4)
    new_image, alpha_channel = split_alpha_channel(image)
    assert len(new_image.shape) == 3
    assert len(alpha_channel.shape) == 3
    assert new_image.shape[-1] == 3
    assert alpha_channel.shape[-1] == 1
    assert np.allclose(image[:, :, :3], new_image)
    assert np.allclose(image[:, :, 3:4], alpha_channel)


def test_sample_uniformly():
    for distance in range(1, 100):
        assert np.isclose(distance, sample_uniformly(distance))
        sample = sample_uniformly([-distance, distance])
        assert -distance < sample < +distance


def test_get_look_at_transform_origin(origin_A, target_A, transform_A):
    transform = get_look_at_transform(origin_A, target_A)
    assert np.allclose(transform, transform_A)


def test_get_look_at_transform_random(origin_B, target_B, transform_B):
    transform = get_look_at_transform(origin_B, target_B)
    assert np.allclose(transform, transform_B)


def test_compute_modelview_matrices_origin(origin_A, target_A, transform_A):
    matrices = compute_modelview_matrices(origin_A, target_A)
    camera_to_world, world_to_camera = matrices
    assert np.allclose(world_to_camera, transform_A)
    assert np.allclose(np.linalg.inv(world_to_camera), camera_to_world)


def test_compute_modelview_matrices_random(origin_B, target_B, transform_B):
    matrices = compute_modelview_matrices(origin_B, target_B)
    camera_to_world, world_to_camera = matrices
    assert np.allclose(world_to_camera, transform_B)
    assert np.allclose(np.linalg.inv(world_to_camera), camera_to_world)


def test_scale_translation_origin(transform_B, distance=10.0):
    transform = scale_translation(transform_B.copy(), distance)
    assert np.allclose(transform_B[:3, -1] * distance, transform[:3, -1])


def test_scale_translation_random(transform_B, distance=10.0):
    transform = scale_translation(transform_B.copy(), distance)
    assert np.allclose(transform_B[:3, -1] * distance, transform[:3, -1])


def test_roll_camera_origin_zero(transform_A):
    transform = roll_camera(transform_A.copy(), 0.0)
    assert np.allclose(transform_A, transform)


def test_roll_camera_random_zero(transform_B):
    transform = roll_camera(transform_B.copy(), 0.0)
    assert np.allclose(transform_B, transform)


def test_roll_camera_origin_axis(transform_A):
    for arg in range(100):
        angle = np.random.uniform(0, 2 * np.pi)
        transform = roll_camera(transform_A.copy(), angle)
        assert np.allclose(transform[2, :3], transform_A[2, :3])


def test_roll_camera_random_axis(transform_B):
    for arg in range(100):
        angle = np.random.uniform(0, 2 * np.pi)
        transform = roll_camera(transform_B.copy(), angle)
        assert np.allclose(transform[2, :3], transform_B[2, :3])


def test_translate_camera_origin_zero(transform_A):
    transform = translate_camera(transform_A.copy(), 0.0)
    assert np.allclose(transform, transform_A)


def test_translate_camera_random_zero(transform_B):
    transform = translate_camera(transform_B.copy(), 0.0)
    assert np.allclose(transform, transform_B)


def test_translate_camera_origin_axis(transform_A):
    for arg in range(100):
        translation = np.random.uniform(-10, 10, 2)
        transform = translate_camera(transform_A.copy(), translation)
        assert np.allclose(transform[:3, :3], transform_A[:3, :3])


def test_translate_camera_random_axis(transform_B):
    for arg in range(100):
        translation = np.random.uniform(-10, 10, 2)
        transform = translate_camera(transform_B.copy(), translation)
        assert np.allclose(transform[:3, :3], transform_B[:3, :3])
