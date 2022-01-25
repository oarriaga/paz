import pytest
import numpy as np

from .weighted_reconstruction import split_alpha_mask
from .weighted_reconstruction import compute_foreground_loss
from .weighted_reconstruction import compute_background_loss
from .weighted_reconstruction import compute_error_prediction_loss
from .weighted_reconstruction import compute_weighted_reconstruction_loss
from .weighted_reconstruction import (
    compute_weighted_reconstruction_loss_with_error)
from .weighted_reconstruction import (
    normalized_image_to_normalized_device_coordinates,
    normalized_device_coordinates_to_normalized_image)
from .weighted_reconstruction import WeightedReconstruction


@pytest.fixture
def RGBA_mask():
    return np.ones((32, 128, 128, 4), dtype=np.float32)


@pytest.fixture
def RGB_true():
    return np.ones((32, 128, 128, 3), dtype=np.float32)


@pytest.fixture
def RGBA_true():
    return np.ones((32, 128, 128, 4), dtype=np.float32)


@pytest.fixture
def RGB_pred():
    return 0.5 * np.ones((32, 128, 128, 3), dtype=np.float32)


@pytest.fixture
def RGBE_pred():
    return 0.5 * np.ones((32, 128, 128, 4), dtype=np.float32)


@pytest.fixture
def alpha_mask():
    return np.ones((32, 128, 128, 1), dtype=np.float32)


def test_split_alpha_mask(RGBA_mask):
    batch_size, H, W, num_channels = RGBA_mask.shape
    color_mask, alpha_mask = split_alpha_mask(RGBA_mask)
    assert color_mask.shape == (batch_size, H, W, 3)
    assert alpha_mask.shape == (batch_size, H, W, 1)


def test_split_error_mask(RGBA_mask):
    batch_size, H, W, num_channels = RGBA_mask.shape
    color_mask, alpha_mask = split_alpha_mask(RGBA_mask)
    assert color_mask.shape == (batch_size, H, W, 3)
    assert alpha_mask.shape == (batch_size, H, W, 1)


def test_compute_foreground_loss(RGB_true, RGB_pred, alpha_mask):
    foreground_loss = compute_foreground_loss(RGB_true, RGB_pred, alpha_mask)
    assert np.allclose(foreground_loss, 0.5)


def test_compute_background_loss(RGB_true, RGB_pred, alpha_mask):
    alpha_mask = 1.0 - alpha_mask
    background_loss = compute_background_loss(RGB_true, RGB_pred, alpha_mask)
    assert np.allclose(background_loss, 0.5)


def test_compute_weighted_reconstruction_loss(RGBA_true, RGB_pred):
    loss = compute_weighted_reconstruction_loss(RGBA_true, RGB_pred, 3.0)
    assert np.allclose(loss, 1.5)


def test_normalized_image_to_normalized_device_coordinates(RGB_true):
    value = normalized_image_to_normalized_device_coordinates(RGB_true)
    assert np.max(value) == 1.0


def test_normalized_image_to_normalized_device_coordinates_segment():
    image = np.array([0, 0.5, 1.0])
    value = normalized_image_to_normalized_device_coordinates(image)
    assert ((np.min(value) == -1.0) and (np.max(value) == 1.0))


def test_normalized_device_coordinates_to_normalized_image():
    image = np.array([-1.0, 0.0, 1.0])
    value = normalized_device_coordinates_to_normalized_image(image)
    assert ((np.min(value) == 0.0) and (np.max(value) == 1.0))


def test_weighted_reconstruction_loss(RGBA_true, RGB_pred):
    compute_loss = WeightedReconstruction(beta=3.0)
    loss = compute_loss(RGBA_true, RGB_pred)
    assert np.allclose(loss, 1.5)


def test_weighted_reconstruction_loss_with_error(RGBA_true, RGBE_pred):
    loss = compute_weighted_reconstruction_loss_with_error(
        RGBA_true, RGBE_pred, beta=3.0)
    assert np.allclose(loss, 1.5)


def test_error_prediction_loss(RGBA_true, RGBE_pred):
    # TODO change RGBE_pred
    loss = compute_error_prediction_loss(RGBA_true, RGBE_pred)
    print(loss)
    assert True

# test_WeightedReconstructionWithError
# test_ErrorPrediction

# test_WeightedSymmetricReconstruction
# test_compute_weighted_symmetric_loss
