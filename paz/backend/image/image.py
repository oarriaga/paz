import numpy as np

from .opencv_image import (convert_color_space, gaussian_image_blur,
                           median_image_blur, warp_affine, resize_image,
                           apply_histogram_equalization, RGB2HSV, HSV2RGB)


def cast_image(image, dtype):
    """Casts an image into a different type

    # Arguments
        image: Numpy array.
        dtype: String or np.dtype.

    # Returns
        Numpy array.
    """
    return image.astype(dtype)


def random_saturation(image, lower=0.3, upper=1.5):
    """Applies random saturation to an RGB image.

    # Arguments
        image: Numpy array representing an image RGB format.
        lower: Float.
        upper: Float.
    """
    image = convert_color_space(image, RGB2HSV)
    image = cast_image(image, np.float32)
    image[:, :, 1] = image[:, :, 1] * np.random.uniform(lower, upper)
    image[:, :, 1] = np.clip(image[:, :, 1], 0, 255)
    image = cast_image(image, np.uint8)
    image = convert_color_space(image, HSV2RGB)
    return image


def random_brightness(image, delta=32):
    """Applies random brightness to an RGB image.

    # Arguments
        image: Numpy array representing an image RGB format.
        delta: Int.
    """
    image = cast_image(image, np.float32)
    random_brightness = np.random.uniform(-delta, delta)
    image = image + random_brightness
    image = np.clip(image, 0, 255)
    image = cast_image(image, np.uint8)
    return image


def random_contrast(image, lower=0.5, upper=1.5):
    """Applies random contrast to an RGB image.

    # Arguments
        image: Numpy array representing an image RGB format.
        lower: Float.
        upper: Float.
    """
    alpha = np.random.uniform(lower, upper)
    image = cast_image(image, np.float32)
    image = image * alpha
    image = np.clip(image, 0, 255)
    image = cast_image(image, np.uint8)
    return image


def random_hue(image, delta=18):
    """Applies random hue to an RGB image.

    # Arguments
        image: Numpy array representing an image RGB format.
        delta: Int.
    """
    image = convert_color_space(image, RGB2HSV)
    image = cast_image(image, np.float32)
    image[:, :, 0] = image[:, :, 0] + np.random.uniform(-delta, delta)
    image[:, :, 0][image[:, :, 0] > 179.0] -= 179.0
    image[:, :, 0][image[:, :, 0] < 0.0] += 179.0
    image = cast_image(image, np.uint8)
    image = convert_color_space(image, HSV2RGB)
    return image


def flip_left_right(image):
    """Flips an image left and right.

    # Arguments
        image: Numpy array.
    """
    return image[:, ::-1]


def random_flip_left_right(image):
    """Applies random left or right flip.

    # Arguments
        image: Numpy array.
    """
    if np.random.uniform([1], 0, 2) == 1:
        image = flip_left_right(image)
    return image


def crop_image(image, crop_box):
    """Resize image.

    # Arguments
        image: Numpy array.
        crop_box: List of four ints.

    # Returns
        Numpy array.
    """
    if (type(image) != np.ndarray):
        raise ValueError(
            'Recieved Image is not of type numpy array', type(image))
    else:
        cropped_image = image[crop_box[0]:crop_box[2],
                              crop_box[1]:crop_box[3], :]
    return cropped_image


def image_to_normalized_device_coordinates(image):
    """Map image value from [0, 255] -> [-1, 1].
    """
    return (image / 127.5) - 1.0


def normalized_device_coordinates_to_image(image):
    """Map normalized value from [-1, 1] -> [0, 255].
    """
    return (image + 1.0) * 127.5


def random_shape_crop(image, shape):
    """Randomly crops an image of the given ``shape``.

    # Arguments
        image: Numpy array.
        shape: List of two ints ''(H, W)''.

    # Returns
        Numpy array of cropped image.
    """
    H, W = image.shape[:2]
    if (shape[0] >= H) or (shape[1] >= W):
        return None
    x_min = np.random.randint(0, W - shape[1])
    y_min = np.random.randint(0, H - shape[0])
    x_max = int(x_min + shape[1])
    y_max = int(y_min + shape[0])
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def make_random_plain_image(shape):
    """Makes random plain image by sampling three random values.

    # Arguments
        shape: Image shape e.g. ''(H, W, 3)''.

    # Returns
        Numpy array of shape ''(H, W, 3)''.
    """
    if len(shape) != 3:
        raise ValueError('``shape`` must have three values')
    return (np.ones(shape) * np.random.randint(0, 256, shape[-1]))


def blend_alpha_channel(image, background):
    """Blends image with background using an alpha channel.

    # Arguments
        image: Numpy array with alpha channel. Shape must be ''(H, W, 4)''
        background: Numpy array of shape ''(H, W, 3)''.
    """
    if image.shape[-1] != 4:
        raise ValueError('``image`` does not contain an alpha mask.')
    foreground, alpha = np.split(image, [3], -1)
    alpha = alpha / 255.0
    background = (1.0 - alpha) * background.astype(float)
    image = (alpha * foreground.astype(float)) + background
    return image.astype('uint8')


def concatenate_alpha_mask(image, alpha_mask):
    """Concatenates alpha mask to image.

    # Arguments
        image: Numpy array of shape ''(H, W, 3)''.
        alpha_mask: Numpy array array of shape ''(H, W)''.

    # Returns
        Numpy array of shape ''(H, W, 4)''.
    """
    return np.concatenate([image, alpha_mask], axis=2)


def split_and_normalize_alpha_channel(image):
    """Splits alpha channel from an RGBA image and normalizes alpha channel.

    # Arguments
        image: Numpy array of shape ''(H, W, 4)''.

    # Returns
        List of two numpy arrays containing respectively the image and the
            alpha channel.
    """
    if image.shape[-1] != 4:
        raise ValueError('Provided image does not contain alpha mask.')
    image, alpha_channel = np.split(image, [3], -1)
    alpha_channel = alpha_channel / 255.0
    return image, alpha_channel


def random_image_blur(image):
    """Applies random choice blur.

    # Arguments
        image: Numpy array of shape ''(H, W, 3)''.

    # Returns
        Numpy array.
    """
    blur = np.random.choice([gaussian_image_blur, median_image_blur])
    return blur(image)


def translate_image(image, translation, fill_color):
    """Translate image.

    # Arguments
        image: Numpy array.
        translation: A list of length two indicating the x,y translation values
        fill_color: List of three floats representing a color.

    # Returns
        Numpy array
    """
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0], matrix[1, 1] = 1.0, 1.0
    matrix[0, 2], matrix[1, 2] = translation
    image = warp_affine(image, matrix, fill_color)
    return image


def sample_scaled_translation(delta_scale, image_shape):
    """Samples a scaled translation from a uniform distribution.

    # Arguments
        delta_scale: List with two elements having the normalized deltas.
            e.g. ''[.25, .25]''.
        image_shape: List containing the height and width of the image.
    """
    x_delta_scale, y_delta_scale = delta_scale
    x = image_shape[1] * np.random.uniform(-x_delta_scale, x_delta_scale)
    y = image_shape[0] * np.random.uniform(-y_delta_scale, y_delta_scale)
    return [x, y]


def replace_lower_than_threshold(source, threshold=1e-3, replacement=0.0):
    """Replace values from source that are lower than the given threshold.
    This function doesn't create a new array but does replacement in place.

    # Arguments
        source: Array.
        threshold: Float. Values lower than this value will be replaced.
        replacement: Float. Value taken by elements lower than threshold.

    # Returns
        Array of same shape as source.
    """
    lower_than_epsilon = source < threshold
    source[lower_than_epsilon] = replacement
    return source


def normalize_min_max(x, x_min, x_max):
    """Normalized data using it's maximum and minimum values

    # Arguments
        x: array
        x_min: minimum value of x
        x_max: maximum value of x

    # Returns
        min-max normalized data
    """
    return (x - x_min) / (x_max - x_min)


def calculate_image_center(image):
    '''
    Return image center.

    # Arguments
        image: Numpy array.

    # Returns
        image center.
    '''
    H, W = image.shape[:2]
    center_W = W / 2.0
    center_H = H / 2.0
    return center_W, center_H


def get_scaling_factor(image, scale=1, shape=(128, 128)):
    '''
    Return scaling factor for the image.

    # Arguments
        image: Numpy array.
        scale: Int.
        shape: Tuple of integers. eg. (128, 128)

    # Returns
        scaling factor: Numpy array of size 2
    '''
    H, W = image.shape[:2]
    H_scale = H / shape[0]
    W_scale = W / shape[1]
    return np.array([W_scale * scale, H_scale * scale])


def scale_resize(image, image_size):
    """Resizes and crops image by returning the scales to original
    image.

    Args:
        image: Numpy array, raw image.
        image_size: Int, size of the image.

    Returns:
        Tuple: output_image, image_scale.
    """
    H, W = image.shape[0], image.shape[1]
    image_scale_x = image_size / W
    image_scale_y = image_size / H
    image_scale = min(image_scale_x, image_scale_y)
    scaled_H = int(H * image_scale)
    scaled_W = int(W * image_scale)
    scaled_image = resize_image(image, (scaled_W, scaled_H))
    scaled_image = scaled_image[:image_size, :image_size, :]
    output_image = np.zeros((image_size, image_size, image.shape[2]))
    output_image[:scaled_image.shape[0],
                 :scaled_image.shape[1],
                 :scaled_image.shape[2]] = scaled_image
    image_scale = np.array(1 / image_scale)
    output_image = output_image[np.newaxis]
    return output_image, image_scale


def compute_resizing_shape(image, size):
    """Computes the final size of the image to be scaled by `size`
    such that the largest dimension of the image is equal to `size`.

    # Arguments
        image: Array, raw image to be scaled.
        size: Int, final size of the image.

    # Returns
        List: Containing final shape of image and scale.
    """
    H, W = image.shape[:2]
    image_scale = size / max(H, W)
    resizing_W = int(W * image_scale)
    resizing_H = int(H * image_scale)
    resizing_shape = (resizing_W, resizing_H)
    return [resizing_shape, np.array(image_scale)]


def pad_image(image, size, mode):
    """Pads the image to the final size `size`.

    # Arguments
        image: Array, image to be padded.
        size: Int, final size of the image.
        mode: Str, specifying the type of padding.

    # Returns
        Array: Padded image.
    """
    H, W = image.shape[:2]
    pad_H = size - H
    pad_W = size - W
    pad_shape = [(0, pad_H), (0, pad_W), (0, 0)]
    return np.pad(image, pad_shape, mode=mode)


def equalize_histogram(image):
    """Performs histogram equalization on a given image.

    # Arguments
        image: Array, raw image.

    # Returns:
        equalized: Array, histogram equalized image.
    """
    R, G, B = np.split(image, 3, axis=2)
    R_equalized = apply_histogram_equalization(R)
    G_equalized = apply_histogram_equalization(G)
    B_equalized = apply_histogram_equalization(B)
    return np.concatenate([R_equalized[:, :, np.newaxis],
                           G_equalized[:, :, np.newaxis],
                           B_equalized[:, :, np.newaxis]], axis=2)


def invert_colors(image):
    """Performs color / gray value inversion on a given image.

    # Arguments
        image: Array, raw image.

    # Returns:
        Array: Color inverted image.
    """
    return 255 - image


def posterize(image, num_bits):
    """Performs posterization on a given image. This is achieved
    by reducing the bit depth of the gray value.

    # Arguments
        image: Array, raw image.
        num_bits: Int, final bit depth after posterization.

    # Returns:
        Array: Posterized image.
    """
    scale_factor = 2 ** (8 - num_bits)
    posterized = np.round(image / scale_factor) * scale_factor
    return posterized.astype(np.uint8)


def solarize(image, threshold):
    """Performs solarization on a given image. This is achieved
    by inverting those pixels whose gray values lie above
    a certain `threshold`.

    # Arguments
        probability: Float, probability of data transformation.
        threshold: Int, threshold value.

    # Returns:
        Array: Solarized image.
    """
    return np.where(image < threshold, image, 255 - image)


def cutout(image, size, fill):
    """Cuts out a square of size `size` x `size` at a random location
    in the `image` and fills it with `fill` value.

    # Arguments
        image: Array, raw image.
        size: Int, size of cutout square.
        fill: Int, value to fill cutout with.

    # Returns:
        image: Array, cutout image.
    """
    H, W, _ = image.shape
    y = np.random.randint(0, H - size)
    x = np.random.randint(0, W - size)
    image[y:y+size, x:x+size, :] = fill
    return image


def add_gaussian_noise(image, mean, sigma):
    """Adds Gaussian noise defined by `mean` and `scale` to the `image`.

    # Arguments
        image: Array, raw image.
        mean: Int, mean of Gaussian noise.
        sigma: Float, standard deviation of Gaussian noise.

    # Returns:
        Array: Image added with Gaussian noise.
    """
    H, W, num_channels = image.shape
    noise = np.random.normal(mean, sigma, (H, W, num_channels))
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)
