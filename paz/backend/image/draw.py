import numpy as np
import colorsys
import random
import cv2

GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
FILLED = cv2.FILLED


def draw_square(image, center, color, size):
    """Draw a square in an image

    # Arguments
        image: Array ``(H, W, 3)``
        center: List ``(2)`` with ``(x, y)`` values in openCV coordinates.
        size: Float. Length of square size.
        color: List ``(3)`` indicating RGB colors.

    # Returns
        Array ``(H, W, 3)`` with square.
    """
    center_x, center_y = center
    x_min, y_min = center_x - size, center_y - size
    x_max, y_max = center_x + size, center_y + size
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), tuple(color), FILLED)
    return image


def draw_circle(image, center, color=GREEN, radius=5):
    """Draw a circle in an image

    # Arguments
        image: Array ``(H, W, 3)``
        center: List ``(2)`` with ``(x, y)`` values in openCV coordinates.
        radius: Float. Radius of circle.
        color: Tuple ``(3)`` indicating the RGB colors.

    # Returns
        Array ``(H, W, 3)`` with circle.
    """
    cv2.circle(image, tuple(center), radius, tuple(color), FILLED)
    return image


def draw_triangle(image, center, color, size):
    """Draw a triangle in an image

    # Arguments
        image: Array ``(H, W, 3)``
        center: List ``(2)`` containing ``(x_center, y_center)``.
        size: Float. Length of square size.
        color: Tuple ``(3)`` indicating the RGB colors.

    # Returns
        Array ``(H, W, 3)`` with triangle.
    """
    center_x, center_y = center
    vertex_A = (center_x, center_y - size)
    vertex_B = (center_x - size, center_y + size)
    vertex_C = (center_x + size, center_y + size)
    points = np.array([[vertex_A, vertex_B, vertex_C]], dtype=np.int32)
    cv2.fillPoly(image, points, tuple(color))
    return image


def draw_keypoint(image, point, color=GREEN, radius=5):
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
    cv2.circle(image, tuple(point), radius, (0, 0, 0), FILLED)
    inner_radius = int(0.8 * radius)
    cv2.circle(image, tuple(point), inner_radius, tuple(color), FILLED)
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
    if points.shape != (8, 2):
        raise ValueError('Cube points 2D must be of shape (8, 2)')

    # draw bottom
    draw_line(image, points[0], points[1], color, thickness)
    draw_line(image, points[1], points[2], color, thickness)
    draw_line(image, points[3], points[2], color, thickness)
    draw_line(image, points[3], points[0], color, thickness)

    # draw top
    draw_line(image, points[4], points[5], color, thickness)
    draw_line(image, points[6], points[5], color, thickness)
    draw_line(image, points[6], points[7], color, thickness)
    draw_line(image, points[4], points[7], color, thickness)

    # draw sides
    draw_line(image, points[0], points[4], color, thickness)
    draw_line(image, points[7], points[3], color, thickness)
    draw_line(image, points[5], points[1], color, thickness)
    draw_line(image, points[2], points[6], color, thickness)

    # draw X mark on top
    draw_line(image, points[4], points[6], color, thickness)
    draw_line(image, points[5], points[7], color, thickness)

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


def draw_points2D(image, points2D, colors):
    """Draws a pixel for all points2D in UV space using only numpy.

    # Arguments
        image: Array (H, W).
        keypoints: Array (num_points, U, V). Keypoints in image space
        colors: Array (num_points, 3). Colors in RGB space.

    # Returns
        Array with drawn points.
    """
    points2D = points2D.astype(int)
    U = points2D[:, 0]
    V = points2D[:, 1]
    image[V, U, :] = colors
    return image


def draw_keypoints_link(image, keypoints, link_args, link_orders, link_colors,
                        check_scores=False, link_width=2):
    """ Draw link between the keypoints.

    # Arguments
        images: Numpy array.
        keypoints: Keypoint(k0, k1, ...) locations in the image. Numpy array.
        link_args: Keypoint labels. Dictionary. {'k0':0, 'k1':1, ...}
        link_orders: List of tuple. [('k0', 'k1'),('kl', 'k2'), ...]
        link_colors: Color of each link. List of list
        check_scores: Condition to draw links. Boolean.

    # Returns
        A numpy array containing drawn link between the keypoints.
    """
    for pair_arg, pair in enumerate(link_orders):
        color = link_colors[pair_arg]
        point1 = keypoints[link_args[pair[0]]]
        point2 = keypoints[link_args[pair[1]]]
        if check_scores:
            if point1[2] > 0 and point2[2] > 0:
                draw_line(image, (int(point1[0]), int(point1[1])),
                                 (int(point2[0]), int(point2[1])),
                          color, link_width)
        else:
            draw_line(image, (int(point1[0]), int(point1[1])),
                             (int(point2[0]), int(point2[1])),
                      color, link_width)
    return image


def draw_keypoints(image, keypoints, keypoint_colors, check_scores=False,
                   keypoint_radius=6):
    """ Draw a circle at keypoints.

    # Arguments
        images: Numpy array.
        keypoints: Keypoint locations in the image. Numpy array.
        keypoint_colors: Color of each keypoint. List of list
        check_scores: Condition to draw keypoint. Boolean.

    # Returns
        A numpy array containing circle at each keypoints.
    """
    for keypoint_arg, keypoint in enumerate(keypoints):
        color = keypoint_colors[keypoint_arg]
        if check_scores:
            if keypoint[2] > 0:
                draw_keypoint(
                    image, (int(keypoint[0]),
                            int(keypoint[1])), color, keypoint_radius)
        else:
            draw_keypoint(image, (int(keypoint[0]), int(keypoint[1])), color,
                          keypoint_radius)
    return image


def points3D_to_RGB(points3D, object_sizes):
    """Transforms points3D in object frame to RGB color space.
    # Arguments
        points3D: Array (num_points, 3). Points3D a
        object_sizes: Array (3) indicating the
            (width, height, depth) of object.

    # Returns
        Array of ints (num_points, 3) in RGB space.
    """
    # TODO add domain and codomain transform as comments
    colors = points3D / (0.5 * object_sizes)
    colors = colors + 1.0
    colors = colors * 127.5
    colors = colors.astype(np.uint8)
    return colors


def draw_RGB_mask(image, points2D, points3D, object_sizes):
    """Draws RGB mask by transforming points3D to RGB space and putting in
        them in their 2D coordinates (points2D)

    # Arguments
        image: Array (H, W, 3).
        points2D: Array (num_points, 2)
        points3D: Array (num_points, 3)
        object_sizes: Array (x_size, y_size, z_size)

    # Returns
        Image array with drawn masks
    """
    color = points3D_to_RGB(points3D, object_sizes)
    image = draw_points2D(image, points2D, color)
    return image


def draw_RGB_masks(image, points2D, points3D, object_sizes):
    """Draws RGB masks by transforming points3D to RGB space and putting in
        them in their 2D coordinates (points2D)

    # Arguments
        image: Array (H, W, 3).
        points2D: Array (num_samples, num_points, 2)
        points3D: Array (num_samples, num_points, 3)
        object_sizes: Array (x_size, y_size, z_size)

    # Returns
        Image array with drawn masks
    """
    for instance_points2D, instance_points3D in zip(points2D, points3D):
        image = draw_RGB_mask(
            image, instance_points2D, instance_points3D, object_sizes)
    return image
