import math
import cv2
import numpy as np
import paz

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA


def rectangle(image, top_left_point, bottom_right_point, color, thickness):
    cv2.rectangle(image, top_left_point, bottom_right_point, color, thickness)
    return image


def box(image, box, color, thickness):
    x_min, y_min, x_max, y_max = box
    return rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)


def boxes(
    image, boxes, class_names, color, box_thickness, font_thickness, font_scale
):
    for box, class_name in zip(boxes, class_names):
        x_min, y_min, x_max, y_max = box
        image = paz.draw.box(image, box, color, box_thickness)

        text_W, text_H = cv2.getTextSize(
            class_name, FONT, font_scale, font_thickness
        )[0]
        text_pad = 5
        text_H = text_H + text_pad

        label_x_min_y_min = (x_min, y_min - text_H)
        label_x_max_y_max = (x_min + text_W, y_min)
        cv2.rectangle(image, label_x_min_y_min, label_x_max_y_max, color, -1)
        image = paz.draw.text(
            image,
            class_name,
            (x_min, y_min - text_pad),
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
    return image


def text(image, text_string, point, scale, color, thickness):
    """Draws text in image.

    # Arguments
        image: Numpy array.
        text_string: String. Text to be drawn.
        point: Tuple of coordinates indicating the top corner of the text.
        scale: Float. Scale of text.
        color: Tuple of integers. RGB color coordinates.
        thickness: Integer. Thickness of the lines used for drawing text.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with text.
    """
    return cv2.putText(
        image, text_string, point, FONT, scale, color, thickness, LINE
    )


def square(image, center, size, color):
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


def circle(image, center, radius, color):
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


def mosaic(images, shape=None, border=0, background=255):
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
        pad = np.full((pad_size, H, W, num_channels), background)
        images = np.concatenate([images, pad], axis=0)
    num_images, H, W, num_channels = images.shape
    total_rows = (num_rows * H) + ((num_rows - 1) * border)
    total_cols = (num_cols * W) + ((num_cols - 1) * border)
    mosaic = np.full((total_rows, total_cols, num_channels), background)

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
