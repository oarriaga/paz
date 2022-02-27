import numpy as np
import pytest
from paz.backend import standard


@pytest.fixture(params=[512])
def input_size(request):
    return (request.param)


@pytest.fixture
def load_image():
    def call(shape, rgb_channel, with_mask=True):
        image = np.ones(shape)
        image[:, :, 0] = rgb_channel[0]
        image[:, :, 1] = rgb_channel[1]
        image[:, :, 2] = rgb_channel[2]
        if with_mask:
            image[10:50, 50:120] = 100
        return image.astype(np.float32)
    return call


@pytest.fixture(params=[(128, 128, 3)])
def image_shape(request):
    return request.param


@pytest.fixture(params=[[50, 120, 201]])
def rgb_channel(request):
    return request.param


@pytest.fixture(params=[200])
def scaling_factor(request):
    return (request.param)


@pytest.fixture(params=[64])
def multiple(request):
    return (request.param)


@pytest.fixture(params=[[32, 57]])
def pixel_location(request):
    return (request.param)


@pytest.fixture(params=[2.25])
def offset(request):
    return request.param


@pytest.fixture
def indices():
    return np.array([[0, 0, 0],
                     [0, 0, 1],
                     [0, 0, 2],
                     [0, 1, 0],
                     [0, 1, 1],
                     [0, 1, 2],
                     [1, 0, 0],
                     [1, 0, 1],
                     [1, 0, 2],
                     [1, 1, 0],
                     [1, 1, 1],
                     [1, 1, 2]])


@pytest.fixture()
def max_pooling_2d_test_matrix():
    return np.array([[0, 1, 2, 4, 5, 6],
                     [3, 4, 5, 6, 9, 2],
                     [6, 7, 8, 6, 13, 4],
                     [14, 27, 4, 8, 37, 3],
                     [6, 8, 9, 27, 24, 13]])


@pytest.fixture()
def valid_max_pooled_2d_matrix():
    return np.array([[8, 8, 13, 13],
                     [27, 27, 37, 37],
                     [27, 27, 37, 37]])


@pytest.fixture()
def same_max_pooled_2d_matrix():
    return np.array([[4, 5, 6, 9, 9, 9],
                     [7, 8, 8, 13, 13, 13],
                     [27, 27, 27, 37, 37, 37],
                     [27, 27, 27, 37, 37, 37],
                     [27, 27, 27, 37, 37, 37]])


def test_get_upper_multiple(multiple):
    upper_multiple = standard.get_upper_multiple(529, 64)
    assert (upper_multiple % multiple == 0)


def test_resize_with_same_aspect_ratio(load_image, rgb_channel, input_size,
                                       image_shape, multiple):
    image = load_image(image_shape, rgb_channel)
    size = standard.resize_with_same_aspect_ratio(image, input_size, multiple)
    assert (size[0] == input_size)
    assert (size[1] % multiple == 0)


@pytest.mark.parametrize("transformation_scale", [[0.64, 0.64]])
def test_get_transformation_scale(load_image, rgb_channel, input_size,
                                  image_shape, scaling_factor, multiple,
                                  transformation_scale):
    image = load_image(image_shape, rgb_channel)
    size = standard.resize_with_same_aspect_ratio(image, input_size, multiple)
    scale = standard.get_transformation_scale(image, size, scaling_factor)
    assert (list(scale) == transformation_scale)


@pytest.mark.parametrize("updated_y_pixel", [54.75])
def test_compare_vertical_neighbors(load_image, rgb_channel, offset,
                                    image_shape, pixel_location,
                                    updated_y_pixel):
    image = load_image(image_shape, rgb_channel)[:, :, 0]
    y = standard.compare_vertical_neighbours(pixel_location[0],
                                             pixel_location[1], image, offset)
    assert (y == updated_y_pixel)


@pytest.mark.parametrize("updated_x_pixel", [29.75])
def test_compare_horizontal_neighbors(load_image, rgb_channel, offset,
                                      image_shape, pixel_location,
                                      updated_x_pixel):
    image = load_image(image_shape, rgb_channel)[:, :, 0]
    x = standard.compare_horizontal_neighbours(
        pixel_location[0], pixel_location[1], image, offset)
    assert (x == updated_x_pixel)


def test_get_all_indices_of_array(load_image, rgb_channel,
                                  image_shape, indices):
    image = load_image(image_shape, rgb_channel)[:2, :2, :]
    all_indices = standard.get_all_indices_of_array(image)
    assert np.allclose(all_indices, indices)


def test_calculate_norm():
    assert np.isclose(1.41421356, standard.calculate_norm([1.0, 1.0]))
    assert np.isclose(1.41421356, standard.calculate_norm([-1.0, 1.0]))
    assert np.isclose(1.41421356, standard.calculate_norm([1.0, -1.0]))
    assert np.isclose(1.41421356, standard.calculate_norm([-1.0, -1.0]))


def test_pad_matrix(image_shape):
    image = np.ones((image_shape[0], image_shape[1]+2))
    valid_pad = standard.pad_matrix(image, pool_size=3,
                                    strides=1, padding='valid')
    square_pad = standard.pad_matrix(image, pool_size=3,
                                     strides=1, padding='square')
    same_pad = standard.pad_matrix(image, pool_size=3,
                                   strides=1, padding='same')
    assert (valid_pad.shape == image.shape)
    assert (square_pad.shape == (image.shape[0]+2, image.shape[1]))
    assert (same_pad.shape == (image.shape[0]+2, image.shape[1]+2))


def test_max_pooling_2d(max_pooling_2d_test_matrix,
                        valid_max_pooled_2d_matrix,
                        same_max_pooled_2d_matrix):
    valid_max_pool = standard.max_pooling_2d(
        max_pooling_2d_test_matrix, pool_size=3, strides=1, padding='valid')
    same_max_pool = standard.max_pooling_2d(
        max_pooling_2d_test_matrix, pool_size=3, strides=1, padding='same')

    assert np.allclose(valid_max_pool, valid_max_pooled_2d_matrix)
    assert np.allclose(same_max_pool, same_max_pooled_2d_matrix)
