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
    """Casts an image into a different type

    # Arguments
        image: Numpy array.
        dtype: String or np.dtype.

    # Returns
        Numpy array.
    """
    return image.astype(dtype)


def resize_image(image, size):
    """Resize image.

    # Arguments
        image: Numpy array.
        dtype: List of two ints.

    # Returns
        Numpy array.
    """
    if(type(image) != np.ndarray):
        raise ValueError(
            'Recieved Image is not of type numpy array', type(image))
    else:
        return cv2.resize(image, size)


def convert_color_space(image, flag):
    """Convert image to a different color space.

    # Arguments
        image: Numpy array.
        flag: PAZ or openCV flag. e.g. paz.backend.image.RGB2BGR.

    # Returns
        Numpy array.
    """
    return cv2.cvtColor(image, flag)


def load_image(filepath, num_channels=3):
    """Load image from a ''filepath''.

    # Arguments
        filepath: String indicating full path to the image.
        num_channels: Int.

    # Returns
        Numpy array.
    """
    image = cv2.imread(filepath, _CHANNELS_TO_FLAG[num_channels])
    image = convert_color_space(image, BGR2RGB)
    return image


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


def show_image(image, name='image', wait=True):
    """Shows RGB image in an external window.

    # Arguments
        image: Numpy array
        name: String indicating the window name.
        wait: Boolean. If ''True'' window stays open until user presses a key.
            If ''False'' windows closes immediately.
    """
    if image.dtype != np.uint8:
        raise ValueError('``image`` must be of type ``uint8``')
    # openCV default color space is BGR
    image = convert_color_space(image, RGB2BGR)
    cv2.imshow(name, image)
    if wait:
        while True:
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
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


def write_image(filepath, image):
    """Writes an image inside ``filepath``. If ``filepath`` doesn't exist
        it makes a directory. If ``image`` has three channels the image is
        converted into BGR and then written. This is done such that this
        function compatible with ``load_image``.

    # Arguments
        filepath: String with image path. It should include postfix e.g. .png
        image: Numpy array.
    """
    directory_name = os.path.dirname(filepath)
    if (not os.path.exists(directory_name) and (len(directory_name) > 0)):
        os.makedirs(directory_name)
    if image.shape[-1] == 3:
        image = convert_color_space(image, RGB2BGR)
    return cv2.imwrite(filepath, image)


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
        print('WARNING: Image is smaller than crop shape', H, W, shape)
        return None
    x_min = np.random.randint(0, (W - 1) - shape[1])
    y_min = np.random.randint(0, (H - 1) - shape[0])
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


def gaussian_image_blur(image, kernel_size=(5, 5)):
    """Applies Gaussian blur to an image.

    # Arguments
        image: Numpy array of shape ''(H, W, 4)''.
        kernel_size: List of two ints e.g. ''(5, 5)''.

    # Returns
        Numpy array
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


def median_image_blur(image, apperture=5):
    """Applies median blur to an image.

    # Arguments
        image: Numpy array of shape ''(H, W, 3)''.
        apperture. Int.

    # Returns
        Numpy array.
    """
    return cv2.medianBlur(image, apperture)


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


def get_rotation_matrix(center, degrees, scale=1.0):
    """Returns a 2D rotation matrix.

    # Arguments
        center: List of two integer values.
        degrees: Float indicating the angle in degrees.

    # Returns
        Numpy array
    """
    return cv2.getRotationMatrix2D(center, degrees, scale)
