import numpy as np
import backend as B
import pytest


@pytest.fixture(params=[512])
def input_size(request):
    return (request.param)


@pytest.fixture(params=[[128, 128, 1]])
def image_shape(request):
    return request.param


@pytest.fixture(params=[1])
def min_scale(request):
    return (request.param)


@pytest.fixture
def load_image():
    def call(shape, intensity_channel):
        image = np.ones(shape)
        image[:, :, 0] = intensity_channel
        return image.astype(np.uint8)
    return call


@pytest.fixture(params=[100])
def intensity_channel(request):
    return request.param


@pytest.fixture(params=[529])
def dims(request):
    return request.param


@pytest.fixture(params=[[2, 1]])
def point2D_a(request):
    return request.param


@pytest.fixture(params=[[1.8, 2.5]])
def point2D_b(request):
    return request.param


@pytest.fixture(params=[(1, 17, 176, 128)])
def output_shape(request):
    return request.param


@pytest.fixture
def load_output():
    def call(shape):
        output = np.ones(shape)
        return output
    return call


@pytest.mark.parametrize("multiple", [64])
def test_get_dims_x64(dims, multiple):
    dims = B.get_dims_x64(dims, 64)
    assert (dims % multiple == 0)


@pytest.mark.parametrize("transformation_size", [[512, 64]])
def test_get_transformation_size(input_size, image_shape, transformation_size):
    size = B.get_transformation_size(
        input_size, image_shape[0], image_shape[1])

    assert (size[0] == transformation_size[0])
    assert (size[1] % transformation_size[1] == 0)


@pytest.mark.parametrize("transformation_scale", [[0.64, 0.64]])
def test_get_transformation_scale(
        input_size, image_shape, transformation_scale):
    size = B.get_transformation_size(input_size, image_shape[0],
                                     image_shape[1])
    scale = B.get_transformation_scale(image_shape[0], size[0], size[1], 200)
    assert (list(scale) == transformation_scale)


@pytest.mark.parametrize("image_center", [[64, 64]])
def test_calculate_image_center(load_image, image_shape, intensity_channel,
                                image_center):
    test_image = load_image(image_shape, intensity_channel)
    center_W, center_H = B.calculate_image_center(test_image)
    assert image_center == [center_W, center_H]


@pytest.mark.parametrize("rotated_point", [np.array([1, -2])])
def test_rotate_point(point2D_a, rotated_point):
    point = B.rotate_point(point2D_a, -90)
    point = (np.array(point)).astype(np.int8)
    assert np.allclose(point, rotated_point)


@pytest.mark.parametrize("third_point", [np.array([3.5, 1.2])])
def test_calculate_third_point(point2D_a, point2D_b, third_point):
    point = B.calculate_third_point(np.array(point2D_a), np.array(point2D_b))
    assert np.allclose(point, third_point)


@pytest.mark.parametrize("resized_output_shape", [(1, 17,  256, 352)])
def test_resize_output(output_shape, load_output, resized_output_shape):
    test_output = load_output(output_shape)
    resized_output = B.resize_output(test_output, (352, 256))
    assert resized_output_shape == resized_output.shape
