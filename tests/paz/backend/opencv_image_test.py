import cv2
import numpy as np

from paz.backend.image import opencv_image

# TODO:
# Add tests for the following random functions:
# random_saturation
# random_brightness
# random_contrast
# random_hue
# random_plain_background
# show_image


def test_cast_image():
    casted_image = opencv_image.cast_image(test_image, dtype=np.float32)
    assert casted_image.dtype == np.float32


def test_resize_image():
    resized_image = opencv_image.resize_image(test_image, (width, height))
    assert resized_image.size == size


def test_convert_color_space():
    converted_colorspace = opencv_image.convert_color_space(
        test_image, cv2.COLOR_RGB2BGR)
    rgb_to_bgr = test_image[..., ::-1]
    assert np.all(converted_colorspace == rgb_to_bgr)


def test_flip_left_right():
    image_filp = opencv_image.flip_left_right(test_image)
    assert np.all(image_filp == flipped_image)


def test_gaussian_blur_output_shape():
    blurred = opencv_image.gaussian_blur(test_image)
    assert test_image.shape == blurred.shape


def test_split_alpha_channel():
    split_alpha_img, alpha_channel = opencv_image.split_alpha_channel(
        masked_image)
    assert np.all(split_alpha_img == test_image)


def test_alpha_blend():
    alpha_blend_image = opencv_image.alpha_blend(
        foreground, background, alpha_mask)
    assert np.all(alpha_blend_image == test_image)


def get_image(shape, r_channel, g_channel, b_channel):
    image = np.ones(shape)
    image[:, :, 0] = r_channel
    image[:, :, 1] = g_channel
    image[:, :, 2] = b_channel
    return image


image_shape = (128, 128, 3)
test_image = get_image(image_shape, 50, 120, 201)
test_image[10:50, 50:120] = 100
test_image = test_image.astype(np.uint8)

alpha_mask = get_image(image_shape, 0, 0, 0)
alpha_mask[10:50, 50:120] = 255
alpha_mask = alpha_mask.astype(float) / 255.

background = get_image(image_shape, 50, 120, 201).astype(float)

foreground = get_image(image_shape, 0, 0, 0)
foreground[10:50, 50:120] = 100
foreground = foreground.astype(float)

b_channel, g_channel, r_channel = cv2.split(test_image)
alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50
masked_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

flipped_image = test_image[:, ::-1]

scale_percent = 60
width = int(test_image.shape[1] * scale_percent / 100)
height = int(test_image.shape[0] * scale_percent / 100)
size = width * height * 3

affine_matrix = np.array([[2., 1., 0.],
                          [0., 3., 4.]])

test_cast_image()
test_resize_image()
test_convert_color_space()
test_flip_left_right()
test_gaussian_blur_output_shape()
test_split_alpha_channel()
test_alpha_blend()
# test_warp_affine()
