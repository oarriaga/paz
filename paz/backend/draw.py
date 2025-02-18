import math
import cv2
import numpy as np

GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)


def square(image, point, size, color):
    """Draw a square in an image

    # Arguments
        image: Array `(H, W, 3)`
        point: List `(2)` with `(x, y)` values in openCV coordinates.
        size: Float. Length of square size.
        color: List `(3)` indicating RGB colors.

    # Returns
        Image array `(H, W, 3)` with square.
    """
    center_x, center_y = point
    x_min = center_x - size
    y_min = center_y - size
    x_max = center_x + size
    y_max = center_y + size
    color = tuple(color)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, cv2.FILLED)
    return image


def rectangle(image, top_left_point, bottom_right_point, color, thickness):
    cv2.rectangle(image, top_left_point, bottom_right_point, color, thickness)
    return image


def box(image, box, color, thickness):
    x_min, y_min, x_max, y_max = box
    return rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)


def circle(image, point, radius, color):
    """Draw a circle in an image

    # Arguments
        image: Array `(H, W, 3)`
        point: List `(2)` with `(x, y)` values in openCV coordinates.
        radius: Float. Radius of circle.
        color: Tuple `(3)` indicating the RGB colors.

    # Returns
        Array `(H, W, 3)` with circle.
    """
    cv2.circle(image, tuple(point), radius, tuple(color), cv2.FILLED)
    return image


def triangle(image, center, size, color):
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


def mosaic(images, shape=None, border=0):
    """Makes a mosaic of the images.
    # Arguments
        images: Array
    """
    if shape is None:
        large_size = math.ceil(math.sqrt(len(images)))
        small_size = round(math.sqrt(len(images)))
        shape = (large_size, small_size)
    num_rows, num_cols = shape
    if num_rows == num_cols == -1:
        raise ValueError(f"Value shape cannot be all -1")
    if num_rows == -1:
        num_rows = math.ceil(len(images) / num_cols)
    if num_cols == -1:
        num_cols = math.ceil(len(images) / num_rows)

    if len(images) > (num_rows * num_cols):
        raise ValueError(
            f"Number of images {len(images)} bigger than mosaic shape {(num_rows, num_cols)}"
        )
    if len(images) < (num_rows * num_cols):
        num_images, H, W, num_channels = images.shape
        pad_size = (num_rows * num_cols) - num_images
        pad = np.full((pad_size, H, W, num_channels), 255)
        images = np.concatenate([images, pad], axis=0)
    num_images, H, W, num_channels = images.shape
    total_rows = (num_rows * H) + ((num_rows - 1) * border)
    total_cols = (num_cols * W) + ((num_cols - 1) * border)
    mosaic = np.full((total_rows, total_cols, num_channels), 255)

    padded_H = H + border
    padded_W = W + border

    for image_arg, image in enumerate(images):
        row = int(np.floor(image_arg / num_cols))
        col = image_arg % num_cols
        mosaic[
            row * padded_H : row * padded_H + H,
            col * padded_W : col * padded_W + W,
            :,
        ] = image
    return mosaic
