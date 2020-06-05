import cv2
import numpy as np

from paz.backend.image import opencv_image

image = np.ones((128, 128, 3))
image[:, :, 0] = 50
image[:, :, 1] = 120
image[:, :, 2] = 201
image[10:25, 30:50] = 0
image = image.astype(np.uint8)

flipped_image = image[:, ::-1]
file_path = '/home/incendio/Pictures/power_drill.png'

scale_percent = 60 
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
size = width * height * 3


def test_cast_image():
    casted_image = opencv_image.cast_image(image, dtype=np.float32)
    assert casted_image.dtype == np.float32

def test_resize_image():
    resized_image = opencv_image.resize_image(image, (width, height))
    assert resized_image.size == size

def test_convert_color_space():
    converted_colorspace = opencv_image.convert_color_space(image, cv2.COLOR_RGB2BGR)
    rgb_to_bgr = image[..., ::-1]
    assert np.all(converted_colorspace) == np.all(rgb_to_bgr)

def test_load_image():
    load_image = opencv_image.load_image(file_path)
    assert np.all(load_image) == np.all(image)

def test_random_saturation():
    saturated_img = opencv_image.random_saturation(image)
    brightness = 40
    image[image < 255-brightness] += brightness

def test_flip_left_right():
    image_filp = opencv_image.flip_left_right(image)
    assert np.all(image_filp) == np.all(flipped_image)

def test_gaussian_blur():
    blurred = opencv_image.gaussian_blur(image)

test_cast_image()
test_resize_image()
test_convert_color_space()
test_load_image()
test_random_saturation()
test_flip_left_right()
test_gaussian_blur()