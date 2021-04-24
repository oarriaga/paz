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


def draw_square(image, center_x, center_y, size, color):
    x_min, y_min = center_x - size, center_y - size
    x_max, y_max = center_x + size, center_y + size
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, -1)
    return image


def draw_circle(image, center_x, center_y, size, color):
    cv2.circle(image, (center_x, center_y), size, color, -1)
    return image


def draw_triangle(image, center_x, center_y, size, color):
    vertex_A = (center_x, center_y - size)
    vertex_B = (center_x - size, center_y + size)
    vertex_C = (center_x + size, center_y + size)
    points = np.array([[vertex_A, vertex_B, vertex_C]], dtype=np.int32)
    cv2.fillPoly(image, points, color)
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
