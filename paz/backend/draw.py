import colorsys
import math
from functools import lru_cache

import jax
import numpy as np
import cv2
import paz

GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
ORANGE = (255, 165, 0)


@lru_cache(maxsize=None)
def _lincolors(num_colors):
    if num_colors == 0:
        return ()
    return tuple(lincolor(num_colors))


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
    if class_args is None:
        class_args = [0] * len(boxes)
    for box, class_arg, score in zip(boxes, class_args, scores):
        draw_box2D(image, box, class_arg, score)
    return image


def keypoint(image, point, color=GREEN, radius=5):
    """Draws a circle in image.

    # Arguments
        image: Numpy array of shape ``[H, W, 3]``.
        point: List of length two indicating ``(y, x)``
            openCV coordinates.
        color: List of length three indicating RGB color of point.
        radius: Integer indicating the radius of the point to be drawn.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with circle.
    """
    cv2.circle(image, tuple(point), radius, (0, 0, 0), cv2.FILLED)
    inner_radius = int(0.8 * radius)
    cv2.circle(image, tuple(point), inner_radius, tuple(color), cv2.FILLED)
    return image


def keypoints(image, points, colors, radius):
    image = np.ascontiguousarray(np.array(image, dtype=image.dtype))
    points = np.array(points, dtype=points.dtype)
    for point, color in zip(points, colors):
        paz.draw.keypoint(image, point, color, radius)
    return image


def boxes_and_points(image, boxes, all_points, box_color, points_colors):
    image = paz.draw.boxes(image, boxes, box_color, thickness=3)
    for points in all_points:
        image = paz.draw.keypoints(image, points, points_colors, 8)
    return image


def axis(image, camera_intrinsics, object_to_camera):

    def to_tuple(x):
        return tuple(x.astype(int).tolist())

    image = paz.to_numpy(image)
    camera_matrix = camera_intrinsics @ object_to_camera
    center = paz.pinhole.project_to_2D(camera_matrix, np.array([0.0, 0.0, 0.0]))
    x_axis = paz.pinhole.project_to_2D(camera_matrix, np.array([0.1, 0.0, 0.0]))
    y_axis = paz.pinhole.project_to_2D(camera_matrix, np.array([0.0, 0.1, 0.0]))
    z_axis = paz.pinhole.project_to_2D(camera_matrix, np.array([0.0, 0.0, 0.1]))
    center = to_tuple(center)
    image = cv2.line(image, center, to_tuple(x_axis), (255, 0, 0), 3)
    image = cv2.line(image, center, to_tuple(y_axis), (0, 255, 0), 3)
    image = cv2.line(image, center, to_tuple(z_axis), (0, 0, 255), 3)
    image = paz.to_jax(image)
    return image


def line(image, point_A, point_B, color=GREEN, thickness=5):
    """Draws a line in image from ``point_A`` to ``point_B``.

    # Arguments
        image: Numpy array of shape ``[H, W, 3]``.
        point_A: List of length two indicating ``(y, x)`` openCV coordinates.
        point_B: List of length two indicating ``(y, x)`` openCV coordinates.
        color: List of length three indicating RGB color of point.
        thickness: Integer indicating the thickness of the line to be drawn.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with line.
    """
    cv2.line(image, tuple(point_A), tuple(point_B), tuple(color), thickness)
    return image


def dot(image, point, color=GREEN, radius=5, filled=cv2.FILLED):
    """Draws a dot (small rectangle) in image.

    # Arguments
        image: Numpy array of shape ``[H, W, 3]``.
        point: List of length two indicating ``(y, x)`` openCV coordinates.
        color: List of length three indicating RGB color of point.
        radius: Integer indicating the radius of the point to be drawn.
        filled: Boolean. If `True` rectangle is filled with `color`.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with dot.
    """
    # drawing outer black rectangle
    point_A = (int(point[0] - radius), int(point[1] - radius))
    point_B = (int(point[0] + radius), int(point[1] + radius))
    rectangle(image, tuple(point_A), tuple(point_B), color, filled)

    # drawing innner rectangle with given `color`
    inner_radius = int(0.8 * radius)
    point_A = (int(point[0] - inner_radius), int(point[1] - inner_radius))
    point_B = (int(point[0] + inner_radius), int(point[1] + inner_radius))
    rectangle(image, tuple(point_A), tuple(point_B), color, filled)
    return image


def cube(image, points, color=GREEN, thickness=2, radius=5):
    """Draws a cube in image.

    # Arguments
        image: Numpy array of shape (H, W, 3).
        points: List of length 8  having each element a list
            of length two indicating (U, V) openCV coordinates.
        color: List of length three indicating RGB color of point.
        thickness: Integer indicating the thickness of the line to be drawn.
        radius: Integer indicating the radius of corner points to be drawn.

    # Returns
        Numpy array with shape (H, W, 3). Image with cube.
    """
    # if points.shape != (8, 2):
    #     raise ValueError("Cube points 2D must be of shape (8, 2)")

    # draw bottom
    line(image, points[0], points[1], color, thickness)
    line(image, points[1], points[2], color, thickness)
    line(image, points[3], points[2], color, thickness)
    line(image, points[3], points[0], color, thickness)

    # draw top
    line(image, points[4], points[5], color, thickness)
    line(image, points[6], points[5], color, thickness)
    line(image, points[6], points[7], color, thickness)
    line(image, points[4], points[7], color, thickness)

    # draw sides
    line(image, points[0], points[4], color, thickness)
    line(image, points[7], points[3], color, thickness)
    line(image, points[5], points[1], color, thickness)
    line(image, points[2], points[6], color, thickness)

    # draw X mark on top
    line(image, points[4], points[6], color, thickness)
    line(image, points[5], points[7], color, thickness)

    # draw dots
    [dot(image, np.squeeze(point), color, radius) for point in points]
    return image


def poses(image, transforms, camera_matrix, thickness=2, radius=4, colors=None):
    def project(camera_matrix, points3D):
        project = jax.vmap(paz.pinhole.project_to_2D, (None, 0))
        return project(camera_matrix, points3D)

    cpu = jax.local_devices(backend="cpu")[0]
    project = jax.jit(project)
    image = np.ascontiguousarray(paz.to_numpy(image))
    transforms = paz.to_numpy(transforms)
    camera_matrix = jax.device_put(camera_matrix, cpu)
    bounds = np.ones(3)
    cube_points3D = paz.pinhole.build_cube_corners(-bounds, bounds)
    cube_points3D = paz.to_numpy(cube_points3D)
    if colors is None:
        colors = _lincolors(len(transforms))
    for color, transform in zip(colors, transforms):
        points3D = paz.algebra.transform_points(transform, cube_points3D)
        points3D = jax.device_put(points3D, cpu)
        points2D = paz.to_numpy(project(camera_matrix, points3D)).astype(int)
        image = paz.draw.cube(image, points2D, color, thickness, radius)
    return image
