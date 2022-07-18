import cv2
import numpy as np
from paz.backend.image.draw import put_text, draw_rectangle
from paz.backend.image.draw import GREEN


def draw_box(image, coordinates, class_name, score,
             color=GREEN, scale=0.7, weighted=False):
    x_min, y_min, x_max, y_max = coordinates
    if weighted:
        color = [int(channel * score) for channel in color]
    text = '{:0.2f}, {}'.format(score, class_name)
    put_text(image, text, (x_min, y_min - 10), scale, color, 1)
    draw_rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    return image


def resize_image_with_nearest_neighbors(image, size):
    """Resize image using nearest neighbors interpolation.

    # Arguments
        image: Numpy array.
        size: List of two ints.

    # Returns
        Numpy array.
    """
    if(type(image) != np.ndarray):
        raise ValueError(
            'Recieved Image is not of type numpy array', type(image))
    else:
        return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
