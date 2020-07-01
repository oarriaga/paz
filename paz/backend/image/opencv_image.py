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
    if(type(image) != np.ndarray):
        raise ValueError("Recieved Image is not of type numpy array", type(image))
    else:
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


def split_alpha_channel(image):
    if image.shape[-1] != 4:
        raise ValueError('Provided image does not contain alpha mask.')
    image, alpha_channel = np.split(image, [3], -1)
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
    random_color = np.random.uniform(0, 255, [3])
    random_color = np.reshape(random_color, [1, 1, 3])
    H, W = image.shape[:2]
    background = np.tile(random_color, [H, W, 1])
    return alpha_blend(image, background, alpha_channel)


def flip_left_right(image):
    return image[:, ::-1]


def random_flip_left_right(image):
    if np.random.uniform([1], 0, 2) == 1:
        image = flip_left_right(image)
    return image


def gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


def median_blur(image, apperture=5):
    return cv2.medianBlur(image, apperture)


def random_image_quality(image):
    blur = np.random.choice([gaussian_blur, median_blur])
    return blur(image)


def random_crop(image, size):
    raise NotImplementedError


def random_cropped_background(image, background):
    image, alpha_channel = split_alpha_channel(image)
    background = random_crop(background, size=image.shape)
    return alpha_blend(image, background, alpha_channel)


def show_image(image, name='image', wait=True):
    """Shows RGB image in an external window.
    # Arguments
        image: Numpy array
        name: String indicating the window name.
    """
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


def save_images(save_path, images):
    """Saves multiple images in a directory
    # Arguments
        save_path: String. Path to directory. If path does not exist it will
        be created.
        images: List of numpy arrays.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for arg, image in enumerate(images):
        save_image(os.path.join(save_path, 'image_%03d.png' % arg), image)
