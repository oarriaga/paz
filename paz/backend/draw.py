import colorsys
import math
import numpy as np
import cv2
import paz

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


def box(image, box, color=GREEN, thickness=2):
    x_min, y_min, x_max, y_max = box[:4]
    return rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)


def boxes(image, boxes, color=GREEN, thickness=2):
    image = np.ascontiguousarray(np.array(image, dtype=image.dtype))
    for box in boxes:
        image = paz.draw.box(image, box.tolist(), color, thickness)
    return image


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


def mosaic(images, shape=None, border=0, background=255):
    """Makes a mosaic of the images.
    # Arguments
        images: Array
    """
    # TODO add dtype check for images. Images should be uint8
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
    mosaic = np.full(
        (total_rows, total_cols, num_channels), background, dtype=images.dtype
    )

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


def lincolor(num_colors, saturation=0.75, value=1.0, normalize=False):
    """Linearly spaced colors in HSV space"""
    colors = []
    hues = [value / num_colors for value in range(0, num_colors)]
    for hue in hues:
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
        if normalize:
            colors.append(rgb_color)
        else:
            r, g, b = rgb_color
            R = int(255 * r)
            G = int(255 * g)
            B = int(255 * b)
            colors.append((R, G, B))
    return colors


def boxes2D(
    image,
    boxes,
    class_args,
    scores,
    names,
    colors,
    thickness=3,
    font_scale=0.7,
    label_color=WHITE,
    font=cv2.FONT_HERSHEY_DUPLEX,
):

    def draw_box2D(image, box, class_arg, score):
        color = colors[class_arg]
        image = paz.draw.box(image, box.tolist(), color, thickness)
        label = f"{names[class_arg]} {score * 100:.0f}%"
        draw_label_box(image, box, label, color)

    def draw_label_box(image, box, label, color):
        text_args = label, font, font_scale, thickness
        (W_text, H_text), baseline = cv2.getTextSize(*text_args)
        offset = round(thickness / 2)
        x_min, y_min, x_max, y_max = box = box.tolist()
        top_left = (x_min - offset, y_min - H_text - baseline - thickness)
        bottom_right = (x_min + W_text, y_min)
        image = rectangle(image, top_left, bottom_right, color, -1)
        bottom_left = (x_min, y_min - baseline)
        cv2.putText(image, label, bottom_left, font, font_scale, label_color)

    image = np.ascontiguousarray(np.array(image, dtype=image.dtype))
    for box, class_arg, score in zip(boxes, class_args, scores):
        draw_box2D(image, box, class_arg, score)
    return image
