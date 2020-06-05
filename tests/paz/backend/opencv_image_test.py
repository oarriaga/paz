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

def test_load_image():
    load_image = opencv_image.load_image(file_path)
    #convert the image to BGR
    assert np.all(load_image == test_image[..., ::-1])

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
    alpha = np.ones((150, 150), dtype=b_channel.dtype) * 50
    alpha_blend_image = opencv_image.alpha_blend(foreground, background, alpha)

def test_random_plain_background():
    random_background_img = opencv_image.random_plain_background(masked_image)

def test_show_image():
    img_ = opencv_image.show_image(test_image)

def get_image(shape, r_channel, g_channel, b_channel):
    image = np.ones(shape)
    image[:, :, 0] = r_channel
    image[:, :, 1] = g_channel
    image[:, :, 2] = b_channel
    image = image.astype(np.uint8)
    return image

def extract_foreground(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    backgroundModel = np.zeros((1, 65), np.float64) 
    foregroundModel = np.zeros((1, 65), np.float64)
    rectangle = (50,50,450,290)
    cv2.grabCut(image_, mask, rectangle,   
            backgroundModel, foregroundModel, 
            3, cv2.GC_INIT_WITH_RECT) 
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    extracted_image = image * mask2[:, :, np.newaxis]
    alpha_blend_image = opencv_image.alpha_blend(extracted_image, backgroundModel, mask2[:, :, np.newaxis])


test_image = get_image((128, 128, 3), 50, 120, 201)
test_image[10:25, 30:50] = 1

foreground = get_image((50, 50, 3), 210, 120, 30)
background = get_image((150, 150, 3), 200, 30, 30)

cv2.imwrite('foreground.png', foreground)
cv2.imwrite('backgroun.png', background)

cv2.imwrite('image.png', test_image)
file_path = 'image.png'

b_channel, g_channel, r_channel = cv2.split(test_image)
alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 
masked_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

flipped_image = test_image[:, ::-1]

scale_percent = 60 
width = int(test_image.shape[1] * scale_percent / 100)
height = int(test_image.shape[0] * scale_percent / 100)
size = width * height * 3

# extract_foreground(test_image)
test_cast_image()
test_resize_image()
test_convert_color_space()
test_load_image()
# test_random_saturation()
test_flip_left_right()
test_gaussian_blur_output_shape()
# test_random_brightness()
test_split_alpha_channel()
test_random_plain_background()
# test_show_image()
test_alpha_blend()