import numpy as np
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


def resize_image(image, shape):
    """ Resizes image.
    # Arguments
        image: Numpy array.
        shape: List of two integer elements indicating new shape.
    """
    return cv2.resize(image, shape)


def convert_color_space(image, flag):
    """Converts image to a different color space
    # Arguments
        image: Numpy array
        flag: OpenCV color flag e.g. cv2.COLOR_BGR2RGB or BGR2RGB
    """
    return cv2.cvtColor(image, flag)


def show_image(image, name='image', wait=True):
    """ Shows image in an external window.
    # Arguments
        image: Numpy array
        name: String indicating the window name.
    """
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
