import numpy as np
import cv2
import os

RGB2BGR = cv2.COLOR_RGB2BGR
BGR2RGB = cv2.COLOR_BGR2RGB
RGB2GRAY = cv2.COLOR_RGB2GRAY
RGB2HSV = cv2.COLOR_RGB2HSV
HSV2RGB = cv2.COLOR_HSV2RGB
_CHANNELS_TO_FLAG = {1: cv2.IMREAD_GRAYSCALE,
                     3: cv2.IMREAD_COLOR,
                     4: cv2.IMREAD_UNCHANGED}


def cast_image(image, dtype):
    return image.astype(dtype)


def resize_image(image, size):
    return cv2.resize(image, size)


def convert_color_space(image, flag):
    return cv2.cvtColor(image, flag)


def load_image(filepath, num_channels=3):
    image = cv2.imread(filepath, _CHANNELS_TO_FLAG[num_channels])
    image = convert_color_space(image, BGR2RGB)
    return image


def random_saturation(image, lower=0.3, upper=1.5):
    image = convert_color_space(image, RGB2HSV)
    image = cast_image(image, np.float32)
    image[:, :, 1] = image[:, :, 1] * np.random.uniform(lower, upper)
    image[:, :, 1] = np.clip(image[:, :, 1], 0, 255)
    image = cast_image(image, np.uint8)
    image = convert_color_space(image, HSV2RGB)
    return image


def random_brightness(image, delta=32):
    image = cast_image(image, np.float32)
    random_brightness = np.random.uniform(-delta, delta)
    image = image + random_brightness
    image = np.clip(image, 0, 255)
    image = cast_image(image, np.uint8)
    return image


def random_contrast(image, lower=0.5, upper=1.5):
    alpha = np.random.uniform(lower, upper)
    image = cast_image(image, np.float32)
    image = image * alpha
    image = np.clip(image, 0, 255)
    image = cast_image(image, np.uint8)
    return image


def random_hue(image, delta=18):
    image = convert_color_space(image, RGB2HSV)
    image = cast_image(image, np.float32)
    image[:, :, 0] = image[:, :, 0] + np.random.uniform(-delta, delta)
    image[:, :, 0][image[:, :, 0] > 179.0] -= 179.0
    image[:, :, 0][image[:, :, 0] < 0.0] += 179.0
    image = cast_image(image, np.uint8)
    image = convert_color_space(image, HSV2RGB)
    return image


def flip_left_right(image):
    return image[:, ::-1]


def random_flip_left_right(image):
    if np.random.uniform([1], 0, 2) == 1:
        image = flip_left_right(image)
    return image


def show_image(image, name='image', wait=True):
    """Shows RGB image in an external window.
    # Arguments
        image: Numpy array
        name: String indicating the window name.
    """
    if image.dtype != np.uint8:
        raise ValueError('``image`` must be of type ``uint8``')
    # openCV default color space is BGR
    image = convert_color_space(image, RGB2BGR)
    cv2.imshow(name, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def warp_affine(image, matrix, fill_color=[0, 0, 0]):
    """ Transforms `image` using an affine `matrix` transformation.
    # Arguments
        image: Numpy array.
        matrix: Numpy array of shape (2,3) indicating affine transformation.
        fill_color: List/tuple representing BGR use for filling empty space.
    """
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image, matrix, (width, height), borderValue=fill_color)


def save_image(filepath, image):
    """Saves an image.
    # Arguments
        filepath: String with image path. It should include postfix e.g. .png
        image: Numpy array.
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    return cv2.imwrite(filepath, image)


def random_image_crop(image, size):
    H, W = image.shape[:2]
    if (size >= H) or (size >= W):
        print('WARNING: Image is smaller than crop size')
        return None
    x_min = np.random.randint(0, (W - 1) - size)
    y_min = np.random.randint(0, (H - 1) - size)
    x_max = int(x_min + size)
    y_max = int(y_min + size)
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def make_random_plain_image(shape):
    if len(shape) != 3:
        raise ValueError('``shape`` must have three values')
    return (np.ones(shape) * np.random.randint(0, 256, shape[-1]))


def blend_alpha_channel(image, background):
    if image.shape[-1] != 4:
        raise ValueError('``image`` does not contain an alpha mask.')
    foreground, alpha = np.split(image, [3], -1)
    return (1.0 - (alpha / 255.0)) * background.astype(float)


def concatenate_alpha_mask(image, alpha_mask):
    alpha_mask = np.expand_dims(alpha_mask, axis=-1)
    return np.concatenate([image, alpha_mask], axis=2)


def split_and_normalize_alpha_channel(image):
    if image.shape[-1] != 4:
        raise ValueError('Provided image does not contain alpha mask.')
    image, alpha_channel = np.split(image, [3], -1)
    alpha_channel = alpha_channel / 255.0
    return image, alpha_channel


def gaussian_image_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


def median_image_blur(image, apperture=5):
    return cv2.medianBlur(image, apperture)


def random_image_blur(image):
    blur = np.random.choice([gaussian_image_blur, median_image_blur])
    return blur(image)
