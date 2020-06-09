import cv2
import numpy as np

from paz.backend.image import opencv_image

def test_cast_image():
    casted_image = opencv_image.cast_image(test_image, dtype=np.float32)
    assert casted_image.dtype == np.float32

def test_resize_image():
    resized_image = opencv_image.resize_image(test_image, (width, height))
    assert resized_image.size == size

def test_convert_color_space():
    converted_colorspace = opencv_image.convert_color_space(test_image, cv2.COLOR_RGB2BGR)
    rgb_to_bgr = test_image[..., ::-1]
    assert np.all(converted_colorspace == rgb_to_bgr)

def test_random_saturation():
    saturated_img = opencv_image.random_saturation(test_image)
    # cv2.imwrite('saturated.png', saturated_img)
    # saturation = np.random.uniform(0.3, 1.5)
    # img_saturation = image
    # img_saturation[:, :, 1] = image[:, :, 1] * saturation
    # image[image < 255-brightness] += brightness
    # assert np.all(saturated_img == img_saturation)
    raise NotImplementedError

def test_flip_left_right():
    image_filp = opencv_image.flip_left_right(test_image)
    assert np.all(image_filp == flipped_image)

def test_gaussian_blur_output_shape():
    blurred = opencv_image.gaussian_blur(test_image)
    assert test_image.shape == blurred.shape

def test_random_brightness():
    bright_img = opencv_image.random_brightness(test_image)
    raise NotImplementedError

def test_random_contrast():
    raise NotImplementedError

def test_random_hue():
    raise NotImplementedError

def test_split_alpha_channel():
    split_alpha_img, alpha_channel_ = opencv_image.split_alpha_channel(masked_image)
    assert np.all(split_alpha_img == test_image)

def test_alpha_blend():
    alpha_blend_image = opencv_image.alpha_blend(foreground, background, alpha_mask)
    assert np.all(alpha_blend_image == test_image)

def test_random_plain_background():
    random_background_img = opencv_image.random_plain_background(masked_image)

def test_show_image():
    img_ = opencv_image.show_image(test_image)

def test_warp_affine():
    rows, cols = test_image.shape[:2]
    warp_image = opencv_image.warp_affine(test_image, affine_matrix)
    test_warp = warp_affine(test_image, affine_matrix, cols, rows)
    assert np.all(warp_image == test_warp)

def warp_affine(image, matrix, width, height):
    dst = np.zeros((width, height, 3), dtype=np.float32)
    oldh, oldw = image.shape[:2]
    for u in range(width):
        for v in range(height):
            x = u * matrix[0, 0] + v * matrix[0, 1] + matrix[0, 2]
            y = u * matrix[1, 0] + v * matrix[1, 1] + matrix[1, 2]
            intx, inty = int(x), int(y)
            if 0 < x < oldw and 0 < y < oldh:
                dst[u, v] = image[intx, inty]
    return dst

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
alpha_mask = alpha_mask.astype(float)/255.

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
# test_random_saturation()
test_flip_left_right()
test_gaussian_blur_output_shape()
# test_random_brightness()
test_split_alpha_channel()
test_random_plain_background()
# test_show_image()
test_alpha_blend()
test_warp_affine()