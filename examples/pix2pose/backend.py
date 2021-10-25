import numpy as np
from paz.backend.image.draw import GREEN
from paz.backend.image import draw_line, draw_dot
import cv2


def homogenous_quaternion_to_rotation_matrix(quaternion):
    # w0, q1, q2, q3 = quaternion
    q1, q2, q3, w0 = quaternion

    r11 = w0**2 + q1**2 - q2**2 - q3**2
    r12 = 2 * ((q1 * q2) - (w0 * q3))
    r13 = 2 * ((w0 * q2) + (q1 * q3))

    r21 = 2 * ((w0 * q3) + (q1 * q2))
    r22 = w0**2 - q1**2 + q2**2 - q3**2
    r23 = 2 * ((q2 * q3) - (w0 * q1))

    r31 = 2 * ((q1 * q3) - (w0 * q2))
    r32 = 2 * ((w0 * q1) + (q2 * q3))
    r33 = w0**2 - q1**2 - q2**2 + q3**2

    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])
    return rotation_matrix


def inhomogenous_quaternion_to_rotation_matrix(q):
    """Transforms quaternion into a rotation matrix
    # Arguments
        q: quarternion, Numpy array of shape ``[4]``
    # Returns
        Numpy array representing a rotation vector having a shape ``[3]``.
    """
    # quaternion
    # q = q[::-1]
    r11 = 1 - (2 * (q[1]**2 + q[2]**2))
    r12 = 2 * (q[0] * q[1] - q[3] * q[2])
    r13 = 2 * (q[3] * q[1] + q[0] * q[2])

    r21 = 2 * (q[0] * q[1] + q[3] * q[2])
    r22 = 1 - (2 * (q[0]**2 + q[2]**2))
    r23 = 2 * (q[1] * q[2] - q[3] * q[0])

    r31 = 2 * (q[0] * q[2] - q[3] * q[1])
    r32 = 2 * (q[3] * q[0] + q[1] * q[2])
    r33 = 1 - (2 * (q[0]**2 + q[1]**2))

    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])

    return rotation_matrix
    # return np.squeeze(rotation_matrix)


def multiply_quaternions(quaternion_0, quaternion_1):
    """Multiplies two quaternions.

    # Reference:
        Code extracted from [here](https://stackoverflow.com/questions/
            39000758/how-to-multiply-two-quaternions-by-python-or-numpy)
    """
    x0, y0, z0, w0 = quaternion_0
    x1, y1, z1, w1 = quaternion_1
    x2 = +(x1 * w0) + (y1 * z0) - (z1 * y0) + (w1 * x0)
    y2 = -(x1 * z0) + (y1 * w0) + (z1 * x0) + (w1 * y0)
    z2 = +(x1 * y0) - (y1 * x0) + (z1 * w0) + (w1 * z0)
    w2 = -(x1 * x0) - (y1 * y0) - (z1 * z0) + (w1 * w0)
    return np.array([x2, y2, z2, w2])


# quaternion = (1 / np.sqrt(30)) * np.array([1, 2, 3, 4])
# theta = np.deg2rad(0)
# quaternion = np.array([1, 0, 0, 0])
# a = homogenous_quaternion_to_rotation_matrix(quaternion)
# quaternion = (1 / np.sqrt(30)) * np.array([2, 3, 4, 1])
# b = inhomogenous_quaternion_to_rotation_matrix(quaternion)

def build_cube_points3D(width, height, depth):
    """ Build the 3D points of a cube in the openCV coordinate system:
                               4--------1
                              /|       /|
                             / |      / |
                            3--------2  |
                            |  8_____|__5
                            | /      | /
                            |/       |/
                            7--------6

                   Z (depth)
                  /
                 /_____X (width)
                 |
                 |
                 Y (height)

    # Arguments
        height: float, height of the 3D box.
        width: float,  width of the 3D box.
        depth: float,  width of the 3D box.

    # Returns
        Numpy array of shape ``(8, 3)'' corresponding to 3D keypoints of a cube
    """
    half_height, half_width, half_depth = height / 2., width / 2., depth / 2.
    point_1 = [+half_width, -half_height, +half_depth]
    point_2 = [+half_width, -half_height, -half_depth]
    point_3 = [-half_width, -half_height, -half_depth]
    point_4 = [-half_width, -half_height, +half_depth]
    point_5 = [+half_width, +half_height, +half_depth]
    point_6 = [+half_width, +half_height, -half_depth]
    point_7 = [-half_width, +half_height, -half_depth]
    point_8 = [-half_width, +half_height, +half_depth]
    return np.array([point_1, point_2, point_3, point_4,
                     point_5, point_6, point_7, point_8])


def _preprocess_image_points2D(image_points2D):
    num_points = len(image_points2D)
    image_points2D = image_points2D.reshape(num_points, 1, 2)
    image_points2D = image_points2D.astype(np.float64)
    image_points2D = np.ascontiguousarray(image_points2D)
    return image_points2D


def solve_PnP_RANSAC(object_points3D, image_points2D, camera_intrinsics,
                     inlier_threshold=5, num_iterations=100):
    image_points2D = _preprocess_image_points2D(image_points2D)
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points3D, image_points2D, camera_intrinsics, None,
        flags=cv2.SOLVEPNP_EPNP, reprojectionError=inlier_threshold,
        iterationsCount=num_iterations)
    if success is False:
        rotation_vector, translation = None, None
    return rotation_vector, translation


def project_to_image(rotation, translation, points3D, camera_intrinsics):
    """Project points3D to image plane using a perspective transformation
    """
    if rotation.shape != (3, 3):
        raise ValueError('Rotation matrix is not of shape (3, 3)')
    if len(translation) != 3:
        raise ValueError('Translation vector is not of length 3')
    if len(points3D.shape) != 2:
        raise ValueError('points3D should have a shape (N, 3)')
    if points3D.shape[1] != 3:
        raise ValueError('points3D should have a shape (N, 3)')
    # TODO missing checks for camera intrinsics conditions
    points3D = np.matmul(rotation, points3D.T).T + translation
    x, y, z = np.split(points3D, 3, axis=1)
    x_focal_length = camera_intrinsics[0, 0]
    y_focal_length = camera_intrinsics[1, 1]
    x_image_center = camera_intrinsics[0, 2]
    y_image_center = camera_intrinsics[1, 2]
    x_points = (x_focal_length * (x / z)) + x_image_center
    y_points = (y_focal_length * (y / z)) + y_image_center
    projected_points2D = np.concatenate([x_points, y_points], axis=1)
    return projected_points2D


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


def replace_lower_than_threshold(source, threshold=1e-3, replacement=0.0):
    lower_than_epsilon = source < threshold
    source[lower_than_epsilon] = replacement
    return source


def arguments_to_image_points2D(row_args, col_args):
    row_args = row_args.reshape(-1, 1)
    col_args = col_args.reshape(-1, 1)
    image_points2D = np.concatenate([col_args, row_args], axis=1)
    return image_points2D


def rotation_vector_to_rotation_matrix(rotation_vector):
    rotation_matrix = np.eye(3)
    cv2.Rodrigues(rotation_vector, rotation_matrix)
    return rotation_matrix
