import cv2
import numpy as np


def square(image, center, color, size):
    """Draw a square in an image

    # Arguments
        image: Array `(H, W, 3)`
        center: List `(2)` with `(x, y)` values in openCV coordinates.
        size: Float. Length of square size.
        color: List `(3)` indicating RGB colors.

    # Returns
        Image array `(H, W, 3)` with square.
    """
    center_x, center_y = center
    x_min = center_x - size
    y_min = center_y - size
    x_max = center_x + size
    y_max = center_y + size
    color = tuple(color)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, cv2.FILLED)
    return image


def circle(image, center, color, radius=5):
    """Draw a circle in an image

    # Arguments
        image: Array `(H, W, 3)`
        center: List `(2)` with `(x, y)` values in openCV coordinates.
        radius: Float. Radius of circle.
        color: Tuple `(3)` indicating the RGB colors.

    # Returns
        Array `(H, W, 3)` with circle.
    """
    cv2.circle(image, tuple(center), radius, tuple(color), cv2.FILLED)
    return image


def triangle(image, center, color, size):
    """Draw a triangle in an image

    # Arguments
        image: Array `(H, W, 3)`
        center: List `(2)` containing `(x_center, y_center)`.
        size: Float. Length of square size.
        color: Tuple `(3)` indicating the RGB colors.

    # Returns
        Array `(H, W, 3)` with triangle.
    """
    center_x, center_y = center
    vertex_A = (center_x, center_y - size)
    vertex_B = (center_x - size, center_y + size)
    vertex_C = (center_x + size, center_y + size)
    points = np.array([[vertex_A, vertex_B, vertex_C]], dtype=np.int32)
    cv2.fillPoly(image, points, tuple(color))
    return image
