import cv2
import os


# same flags as in openCV
RGB2BGR = cv2.COLOR_RGB2BGR
BGR2RGB = cv2.COLOR_BGR2RGB
RGB2GRAY = cv2.COLOR_RGB2GRAY
RGB2HSV = cv2.COLOR_RGB2HSV
HSV2RGB = cv2.COLOR_HSV2RGB

_channel_to_flag = {1: cv2.IMREAD_GRAYSCALE,
                    3: cv2.IMREAD_COLOR,
                    4: cv2.IMREAD_UNCHANGED}


def cast_image(image, dtype):
    return image.astype(dtype)


def load_image(filepath, num_channels=3):
    flag = _channel_to_flag[num_channels]
    image = cv2.imread(filepath, flag)
    if num_channels == 3:
        image = convert_color_space(image, BGR2RGB)
    return image


def resize(image, size):
    return cv2.resize(image, size)


def save_image(filepath, image):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    return cv2.imwrite(filepath, image)


def save_image(filepath, images):


def random_saturation(image, upper, lower):
    return tf.image.random_saturation(image, lower, upper)


def random_brightness(image, max_delta):
    return tf.image.random_brightness(image, max_delta)


def random_contrast(image, lower, upper):
    return tf.image.random_contrast(image, lower, upper)


def random_hue(image, max_delta):
    return tf.image.random_hue(image, max_delta)


def random_image_quality(image, lower, upper):
    return tf.image.random_jpeg_quality(image, lower, upper)


def convert_color_space(image, flag):
    return cv2.cvtColor(image, flag)
