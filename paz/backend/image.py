import cv2
import jax
import jax.numpy as jp

# import numpy as np


BILINEAR = cv2.INTER_LINEAR


def flip_left_right(image):
    """Flips an image left and right.

    # Arguments
        image: Array of shape `(H, W, C)`.

    # Returns
        Flipped image array.
    """
    return image[:, ::-1]


def load(filepath):
    image_BGR = cv2.imread(filepath)
    image_RGB = image_BGR[..., ::-1]
    return jp.array(image_RGB)


def resize(image, shape):
    return jax.image.resize(image, (*shape, 3), "bilinear")
