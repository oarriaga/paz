import numpy as np
import os
from tensorflow.keras.utils import get_file
import pytest

from paz.backend import image as opencv_image
from paz.backend.image import split_and_normalize_alpha_channel


@pytest.fixture
def load_image():
    def call(shape, rgb_channel, with_mask=True):
        image = np.ones(shape)
        image[:, :, 0] = rgb_channel[0]
        image[:, :, 1] = rgb_channel[1]
        image[:, :, 2] = rgb_channel[2]
        if with_mask:
            image[10:50, 50:120] = 100
        return image.astype(np.uint8)
    return call


@pytest.fixture(params=[(128, 128, 3)])
def image_shape(request):
    return request.param


@pytest.fixture(params=[[50, 120, 201]])
def rgb_channel(request):
    return request.param


@pytest.fixture(params=[60])
def resized_shape(request):
    def call(image):
        width = int(image.shape[1] * request.param / 100)
        height = int(image.shape[0] * request.param / 100)
        size = width * height * 3
        return (width, height, size)
    return call


@pytest.mark.parametrize('rgb_channel', [[50, 120, 201]])
def test_cast_image(load_image, image_shape, rgb_channel):
    test_image = load_image(image_shape, rgb_channel)
    casted_image = opencv_image.cast_image(test_image, dtype=np.float32)
    assert casted_image.dtype == np.float32


def test_resize_image(load_image, image_shape, rgb_channel, resized_shape):
    test_image = load_image(image_shape, rgb_channel)
    width, height, size = resized_shape(test_image)
    resized_image = opencv_image.resize_image(test_image, (width, height))
    assert resized_image.size == size


def test_convert_color_space(load_image, image_shape, rgb_channel):
    test_image = load_image(image_shape, rgb_channel)
    converted_colorspace = opencv_image.convert_color_space(
        test_image, opencv_image.RGB2BGR)
    rgb_to_bgr = test_image[..., ::-1]
    assert np.all(converted_colorspace == rgb_to_bgr)


def test_flip_left_right(load_image, image_shape, rgb_channel):
    test_image = load_image(image_shape, rgb_channel)
    image_filp = opencv_image.flip_left_right(test_image)
    flipped_image = test_image[:, ::-1]
    assert np.all(image_filp == flipped_image)


def test_gaussian_blur_output_shape(load_image, image_shape, rgb_channel):
    test_image = load_image(image_shape, rgb_channel)
    blurred = opencv_image.gaussian_image_blur(test_image)
    assert test_image.shape == blurred.shape


def test_split_alpha_channel(load_image, image_shape, rgb_channel):
    test_image = load_image(image_shape, rgb_channel)
    b_channel = test_image[:, :, 0]
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50
    masked_image = np.dstack((test_image, alpha_channel))
    split_alpha_img, alpha_mode = split_and_normalize_alpha_channel(
        masked_image)
    assert np.all(split_alpha_img == test_image)


@pytest.mark.parametrize('rgb', [[50, 120, 201]])
def test_alpha_blend(load_image, image_shape, rgb):
    test_image = load_image(image_shape, rgb)
    background_image = load_image(image_shape, rgb,
                                  with_mask=False).astype(float)
    alpha_channel = load_image(image_shape, [0, 0, 0],
                               with_mask=False)
    alpha_channel = alpha_channel[:, :, 0]
    alpha_channel[10:50, 50:120] = 255
    image = np.dstack((test_image, alpha_channel))
    alpha_blend_image = opencv_image.blend_alpha_channel(
        image, background_image)
    assert np.all(alpha_blend_image == test_image)


@pytest.fixture
def image_with_face_fullpath():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.9.1/image_with_face.jpg')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    return fullpath


def test_load_image_shape_with_1_channel(image_with_face_fullpath):
    image = opencv_image.load_image(image_with_face_fullpath, 1)
    assert len(image.shape) == 2


def test_load_image_shape_with_3_channels(image_with_face_fullpath):
    image = opencv_image.load_image(image_with_face_fullpath, 3)
    assert len(image.shape) == 3
    assert image.shape[-1] == 3


def test_load_image_shape_with_4_channels(image_with_face_fullpath):
    image = opencv_image.load_image(image_with_face_fullpath, 4)
    assert len(image.shape) == 3
    assert image.shape[-1] == 4


def test_passes(image_with_face_fullpath):
    with pytest.raises(Exception):
        opencv_image.load_image(image_with_face_fullpath, 2)
        opencv_image.load_image(image_with_face_fullpath, 5)


@pytest.mark.parametrize("image_center", [[64, 64]])
def test_calculate_image_center(load_image, image_shape, rgb_channel,
                                image_center):
    test_image = load_image(image_shape, rgb_channel)
    center_W, center_H = opencv_image.calculate_image_center(test_image)
    assert image_center == [center_W, center_H]
