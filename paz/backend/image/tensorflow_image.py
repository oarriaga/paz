import tensorflow as tf

# same flags as in openCV
RGB2BGR = 4
BGR2RGB = 4
RGB2GRAY = 7
RGB2HSV = 41
HSV2RGB = 55


def cast_image(image, dtype):
    return tf.image.convert_image_dtype(image, dtype)


def load_image(filepath, num_channels=3):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, num_channels, expand_animations=False)
    return image


def resize(image, size):
    return tf.image.resize(image, size)


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


def _RGB_to_grayscale(image):
    return tf.image.rgb_to_grayscale(image)


def _RGB_to_HSV(image):
    return tf.image.rgb_to_hsv(image)


def _HSV_to_RGB(image):
    return tf.image.hsv_to_rgb(image)


def _reverse_channels(image):
    channels = tf.unstack(image, axis=-1)
    image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
    return image


def convert_color_space(image, flag):
    if flag == RGB2BGR:
        image = _reverse_channels(image)

    elif flag == BGR2RGB:
        image = _reverse_channels(image)

    elif flag == RGB2GRAY:
        image = _RGB_to_grayscale(image)

    elif flag == RGB2HSV:
        image = _RGB_to_HSV(image)

    elif flag == HSV2RGB:
        image = _HSV_to_RGB(image)

    elif flag == RGB2HSV:
        image = _RGB_to_HSV(image)

    else:
        raise ValueError('Invalid flag transformation:', flag)

    return image


def random_crop(image, size):
    return tf.image.random_crop(image, size)


def split_alpha_channel(image):
    if image.shape[-1] != 4:
        raise ValueError('Provided image does not contain alpha mask.')
    image, alpha_channel = tf.split(image, [3], -1)
    alpha_channel = alpha_channel / 255.0
    return image, alpha_channel


def alpha_blend(foreground, background, alpha_channel):
    return (alpha_channel * foreground) + ((1.0 - alpha_channel) * background)


def random_plain_background(image):
    """Adds random plain background to image using a normalized alpha channel
    # Arguments
        image: Float array-like with shape (H, W, 4).
        alpha_channel: Float array-like. Normalized alpha channel for blending.
    """
    image, alpha_channel = split_alpha_channel(image)
    random_color = tf.random.uniform([3], 0, 255)
    random_color = tf.reshape(random_color, [1, 1, 3])
    H, W = image.shape[:2]
    background = tf.tile(random_color, [H, W, 1])
    return alpha_blend(image, background, alpha_channel)


def random_cropped_background(image, background):
    image, alpha_channel = split_alpha_channel(image)
    background = random_crop(background, size=image.shape)
    return alpha_blend(image, background, alpha_channel)


def flip_left_right(image):
    return tf.image.flip_left_right(image)


def random_flip_left_right(image):
    if tf.random.uniform([1], 0, 2) == 1:
        image = flip_left_right(image)
    return image


def imagenet_preprocess_input(image, data_format=None, mode='torch'):
    image = tf.keras.applications.imagenet_utils.preprocess_input(image,
                                                                  data_format,
                                                                  mode)
    return image
