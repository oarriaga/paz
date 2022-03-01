import numpy as np
import cv2
import os

RGB2BGR = cv2.COLOR_RGB2BGR
BGR2RGB = cv2.COLOR_BGR2RGB
BGRA2RGBA = cv2.COLOR_BGRA2RGBA
RGB2GRAY = cv2.COLOR_RGB2GRAY
RGB2HSV = cv2.COLOR_RGB2HSV
HSV2RGB = cv2.COLOR_HSV2RGB
_CHANNELS_TO_FLAG = {1: cv2.IMREAD_GRAYSCALE,
                     3: cv2.IMREAD_COLOR,
                     4: cv2.IMREAD_UNCHANGED}
CUBIC = cv2.INTER_CUBIC
BILINEAR = cv2.INTER_LINEAR


def resize_image(image, size, method=BILINEAR):
    """Resize image.

    # Arguments
        image: Numpy array.
        size: List of two ints.
        method: Flag indicating interpolation method i.e.
            paz.backend.image.CUBIC

    # Returns
        Numpy array.
    """
    if (type(image) != np.ndarray):
        raise ValueError(
            'Recieved Image is not of type numpy array', type(image))
    else:
        return cv2.resize(image, size, interpolation=method)


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
    if num_channels not in [1, 3, 4]:
        raise ValueError('Invalid number of channels')

    image = cv2.imread(filepath, _CHANNELS_TO_FLAG[num_channels])
    if num_channels == 3:
        image = convert_color_space(image, BGR2RGB)
    elif num_channels == 4:
        image = convert_color_space(image, BGRA2RGBA)
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


def warp_affine(image, matrix, fill_color=[0, 0, 0], size=None):
    """ Transforms `image` using an affine `matrix` transformation.

    # Arguments
        image: Numpy array.
        matrix: Numpy array of shape (2,3) indicating affine transformation.
        fill_color: List/tuple representing BGR use for filling empty space.
    """
    if size is not None:
        width, height = size
    else:
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


def get_rotation_matrix(center, degrees, scale=1.0):
    """Returns a 2D rotation matrix.

    # Arguments
        center: List of two integer values.
        degrees: Float indicating the angle in degrees.

    # Returns
        Numpy array
    """
    return cv2.getRotationMatrix2D(center, degrees, scale)


def get_affine_transform(source_points, destination_points):
    '''
    Return the transformation matrix.

    # Arguments
        source_points: Numpy array.
        destination_points: Numpy array.

    # Returns
        Transformation matrix.
    '''
    return cv2.getAffineTransform(source_points, destination_points)
