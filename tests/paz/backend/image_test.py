import numpy as np
import pytest

from paz.backend.image import replace_lower_than_threshold
from paz.backend.image import image_to_normalized_device_coordinates
from paz.backend.image import normalized_device_coordinates_to_image
from paz.backend.image import normalize_min_max
from paz.backend.image import get_scaling_factor


def test_replace_lower_than_threshold():
    source = np.ones((128, 128, 3))
    target = replace_lower_than_threshold(source, 2.0, 5.0)
    assert np.allclose(target, 5.0)

    source = np.ones((128, 128, 3))
    target = replace_lower_than_threshold(source, 0.0, -1.0)
    assert np.allclose(target, 1.0)


def test_image_to_normalized_device_coordinates():
    image = np.array([[0, 127.5, 255]])
    values = image_to_normalized_device_coordinates(image)
    assert np.allclose(values, np.array([[-1.0, 0.0, 1.0]]))


def test_normalized_device_coordinates_to_image():
    coordinates = np.array([[-1.0, 0.0, 1.0]])
    values = normalized_device_coordinates_to_image(coordinates)
    assert np.allclose(values, np.array([[0.0, 127.5, 255.0]]))


def test_normalize_min_max():
    x = np.array([-1.0, 0.0, 1.0])
    values = normalize_min_max(x, np.min(x), np.max(x))
    assert np.allclose(values, np.array([0.0, 0.5, 1.0]))


@pytest.mark.parametrize("output_scaling_factor", [[12, 8]])
def test_get_scaling_factor(output_scaling_factor, scale=2, shape=(128, 128)):
    image = np.ones((512, 768, 3))
    scaling_factor = get_scaling_factor(image, scale, shape)
    assert np.allclose(output_scaling_factor, scaling_factor)
