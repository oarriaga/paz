import numpy as np
import colorsys
import random
import cv2

GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
FILLED = cv2.FILLED

def draw_circle(image, point, color=GREEN, radius=5):
    """ Draws a circle in image.

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
    inner_radius = int(.8 * radius)
    # color = color[::-1]  # transform to BGR for openCV
    cv2.circle(image, tuple(point), inner_radius, tuple(color), cv2.FILLED)
    return image


def put_text(image, text, point, scale, color, thickness):
    """Draws text in image.

    # Arguments
        image: Numpy array.
        text: String. Text to be drawn.
        point: Tuple of coordinates indicating the top corner of the text.
        scale: Float. Scale of text.
        color: Tuple of integers. RGB color coordinates.
        thickness: Integer. Thickness of the lines used for drawing text.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with text.
    """
    # cv2.putText returns an image in contrast to other drawing cv2 functions.
    return cv2.putText(image, text, point, FONT, scale, color, thickness, LINE)


def draw_line(image, point_A, point_B, color=GREEN, thickness=5):
    """ Draws a line in image from ``point_A`` to ``point_B``.

    # Arguments
        image: Numpy array of shape ``[H, W, 3]``.
        point_A: List of length two indicating ``(y, x)`` openCV coordinates.
        point_B: List of length two indicating ``(y, x)`` openCV coordinates.
        color: List of length three indicating RGB color of point.
        thickness: Integer indicating the thickness of the line to be drawn.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with line.
    """
    # color = color[::-1]  # transform to BGR for openCV
    cv2.line(image, tuple(point_A), tuple(point_B), tuple(color), thickness)
    return image


def draw_rectangle(image, corner_A, corner_B, color, thickness):
    """ Draws a filled rectangle from ``corner_A`` to ``corner_B``.

    # Arguments
        image: Numpy array of shape ``[H, W, 3]``.
        corner_A: List of length two indicating ``(y, x)`` openCV coordinates.
        corner_B: List of length two indicating ``(y, x)`` openCV coordinates.
        color: List of length three indicating RGB color of point.
        thickness: Integer/openCV Flag. Thickness of rectangle line.
            or for filled use cv2.FILLED flag.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with rectangle.
    """
    # color = color[::-1]  # transform to BGR for openCV
    return cv2.rectangle(
        image, tuple(corner_A), tuple(corner_B), tuple(color), thickness)


def draw_dot(image, point, color=GREEN, radius=5, filled=FILLED):
    """ Draws a dot (small rectangle) in image.

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
    draw_rectangle(image, tuple(point_A), tuple(point_B), color, filled)

    # drawing innner rectangle with given `color`
    inner_radius = int(0.8 * radius)
    point_A = (int(point[0] - inner_radius), int(point[1] - inner_radius))
    point_B = (int(point[0] + inner_radius), int(point[1] + inner_radius))
    draw_rectangle(image, tuple(point_A), tuple(point_B), color, filled)
    return image


def draw_cube(image, points, color=GREEN, thickness=2, radius=5):
    """ Draws a cube in image.

    # Arguments
        image: Numpy array of shape ``[H, W, 3]``.
        points: List of length 8  having each element a list
            of length two indicating ``(y, x)`` openCV coordinates.
        color: List of length three indicating RGB color of point.
        thickness: Integer indicating the thickness of the line to be drawn.
        radius: Integer indicating the radius of corner points to be drawn.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with cube.
    """
    # color = color[::-1]  # transform to BGR for openCV

    # draw bottom
    draw_line(image, points[0][0], points[1][0], color, thickness)
    draw_line(image, points[1][0], points[2][0], color, thickness)
    draw_line(image, points[3][0], points[2][0], color, thickness)
    draw_line(image, points[3][0], points[0][0], color, thickness)

    # draw top
    draw_line(image, points[4][0], points[5][0], color, thickness)
    draw_line(image, points[6][0], points[5][0], color, thickness)
    draw_line(image, points[6][0], points[7][0], color, thickness)
    draw_line(image, points[4][0], points[7][0], color, thickness)

    # draw sides
    draw_line(image, points[0][0], points[4][0], color, thickness)
    draw_line(image, points[7][0], points[3][0], color, thickness)
    draw_line(image, points[5][0], points[1][0], color, thickness)
    draw_line(image, points[2][0], points[6][0], color, thickness)

    # draw X mark on top
    draw_line(image, points[4][0], points[6][0], color, thickness)
    draw_line(image, points[5][0], points[7][0], color, thickness)

    # draw dots
    [draw_dot(image, np.squeeze(point), color, radius) for point in points]
    return image


def draw_filled_polygon(image, vertices, color):
    """ Draws filled polygon

    # Arguments
        image: Numpy array.
        vertices: List of elements each having a list
            of length two indicating ``(y, x)`` openCV coordinates.
        color: Numpy array specifying RGB color of the polygon.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with polygon.
    """
    # color = color[::-1]  # transform to BGR for openCV
    cv2.fillPoly(image, [vertices], color)
    return image


def draw_random_polygon(image, max_radius_scale=.5):
    """Draw random polygon image.

    # Arguments
        image: Numpy array with shape ``[H, W, 3]``.
        max_radius_scale: Float between [0, 1].

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with polygon.
    """
    height, width = image.shape[:2]
    max_distance = np.max((height, width)) * max_radius_scale
    num_vertices = np.random.randint(3, 7)
    angle_between_vertices = 2 * np.pi / num_vertices
    initial_angle = np.random.uniform(0, 2 * np.pi)
    center = np.random.rand(2) * np.array([width, height])
    vertices = np.zeros((num_vertices, 2), dtype=np.int32)
    for vertex_arg in range(num_vertices):
        angle = initial_angle + (vertex_arg * angle_between_vertices)
        vertex = np.array([np.cos(angle), np.sin(angle)])
        vertex = np.random.uniform(0, max_distance) * vertex
        vertices[vertex_arg] = (vertex + center).astype(np.int32)
    color = np.random.randint(0, 256, 3).tolist()
    draw_filled_polygon(image, vertices, color)
    return image


def lincolor(num_colors, saturation=1, value=1, normalized=False):
    """Creates a list of RGB colors linearly sampled from HSV space with
        randomised Saturation and Value.

    # Arguments
        num_colors: Int.
        saturation: Float or `None`. If float indicates saturation.
            If `None` it samples a random value.
        value: Float or `None`. If float indicates value.
            If `None` it samples a random value.
        normalized: Bool. If True, RGB colors are returned between [0, 1]
            if False, RGB colors are between [0, 255].

    # Returns
        List, for which each element contains a list with RGB color
    """
    RGB_colors = []
    hues = [value / num_colors for value in range(0, num_colors)]
    for hue in hues:

        if saturation is None:
            saturation = random.uniform(0.6, 1)

        if value is None:
            value = random.uniform(0.5, 1)

        RGB_color = colorsys.hsv_to_rgb(hue, saturation, value)
        if not normalized:
            RGB_color = [int(color * 255) for color in RGB_color]
        RGB_colors.append(RGB_color)
    return RGB_colors


def make_mosaic(images, shape, border=0):
    """ Creates an image mosaic.

    # Arguments
        images: Numpy array of shape (num_images, height, width, num_channels)
        shape: List of two integers indicating the mosaic shape.
            Shape must satisfy: shape[0] * shape[1] == len(images).
        border: Integer indicating the border per image.

    # Returns
        A numpy array containing all images.
    """
    num_images = len(images)
    num_rows, num_cols = shape
    H, W, num_channels = images.shape[1:]
    mosaic = np.ma.masked_all(
        (num_rows * H + (num_rows - 1) * border,
         num_cols * W + (num_cols - 1) * border, num_channels),
        dtype=np.float32)
    padded_H = H + border
    padded_W = W + border
    for image_arg in range(num_images):
        row = int(np.floor(image_arg / num_cols))
        col = image_arg % num_cols
        image = images[image_arg]
        image_shape = image.shape
        mosaic[row * padded_H:row * padded_H + image_shape[0],
               col * padded_W:col * padded_W + image_shape[1], :] = image
    return mosaic.astype('uint8')
