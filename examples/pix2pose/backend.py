from collections import Iterable
import numpy as np
from paz.backend.image.draw import GREEN
from paz.backend.image import draw_line, draw_dot, draw_circle
from paz.abstract import Pose6D
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


def quaternion_to_rotation_matrix(quaternion, homogenous=True):
    if homogenous:
        matrix = homogenous_quaternion_to_rotation_matrix(quaternion)
    else:
        matrix = inhomogenous_quaternion_to_rotation_matrix(quaternion)
    return matrix


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
    if ((len(object_points3D) < 4) or (len(image_points2D) < 4)):
        raise ValueError('Solve PnP requires at least 4 3D and 2D points')
    image_points2D = _preprocess_image_points2D(image_points2D)
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points3D, image_points2D, camera_intrinsics, None,
        flags=cv2.SOLVEPNP_EPNP, reprojectionError=inlier_threshold,
        iterationsCount=num_iterations)
    translation = np.squeeze(translation, 1)
    return success, rotation_vector, translation


def apply_affine_transform(affine_matrix, vectors):
    return np.matmul(affine_matrix, vectors.T).T


def project_to_image2(affine_matrix, points3D, camera_intrinsics):
    """Project points3D to image plane using a perspective transformation
    """
    if affine_matrix.shape != (4, 4):
        raise ValueError('Affine matrix is not of shape (4, 4)')
    if len(points3D.shape) != 2:
        raise ValueError('points3D should have a shape (N, 3)')
    if points3D.shape[1] != 3:
        raise ValueError('points3D should have a shape (N, 3)')
    # TODO missing checks for camera intrinsics conditions
    points3D = apply_affine_transform(affine_matrix, points3D)
    # points3D = np.matmul(rotation, points3D.T).T + translation
    x, y, z = np.split(points3D, 3, axis=1)
    x_focal_length = camera_intrinsics[0, 0]
    y_focal_length = camera_intrinsics[1, 1]
    x_image_center = camera_intrinsics[0, 2]
    y_image_center = camera_intrinsics[1, 2]
    x_points = (x_focal_length * (x / z)) + x_image_center
    y_points = (y_focal_length * (y / z)) + y_image_center
    projected_points2D = np.concatenate([x_points, y_points], axis=1)
    return projected_points2D


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


def draw_keypoints(image, keypoints, colors, radius):
    for keypoint, color in zip(keypoints, colors):
        R, G, B = color
        color = (int(R), int(G), int(B))
        draw_circle(image, keypoint.astype('int'), color, radius)
    return image


def draw_masks(image, points):
    for points2D, points3D in points:
        object_sizes = np.array([0.184, 0.187, 0.052])
        colors = points3D / (object_sizes / 2.0)
        colors = (colors + 1.0) * 127.5
        colors = colors.astype('int')
        image = draw_maski(image, points2D, colors)
    return image


def draw_maski(image, keypoints, colors, radius=1):
    for keypoint, color in zip(keypoints, colors):
        R, G, B = color
        color = (int(R), int(G), int(B))
        x, y = keypoint
        x = int(x)
        y = int(y)
        draw_dot(image, (x, y), color, radius)
    return image


def rotation_matrix_to_quaternion(rotation_matrix):
    qw = np.sqrt(1 + np.trace(rotation_matrix)) / 2.0

    m21 = rotation_matrix[2, 1]
    m12 = rotation_matrix[1, 2]

    m02 = rotation_matrix[0, 2]
    m20 = rotation_matrix[2, 0]

    m10 = rotation_matrix[1, 0]
    m01 = rotation_matrix[0, 1]

    qx = (m21 - m12) / (4.0 * qw)
    qy = (m02 - m20) / (4.0 * qw)
    qz = (m10 - m01) / (4.0 * qw)
    return qx, qy, qz, qw


def to_pose6D(quaternion, translation, class_name=None):
    return Pose6D(quaternion, translation, class_name)


class MultiList(Iterable):
    def __init__(self, num_lists):
        self.num_lists = num_lists
        self.lists = [[] for list_arg in range(self.num_lists)]

    def append(self, *args):
        if len(args) != self.num_lists:
            raise ValueError('Arguments should have equal lenght as num_lists')
        for arg, arg_list in zip(args, self.lists):
            arg_list.append(arg)

    def __iter__(self):
        return iter(self.lists)


def draw_mask2(image, points3D, object_sizes):
    if len(object_sizes) != 3:
        raise ValueError('Object sizes must contain 3 values')
    colors = points3D / (object_sizes / 2.0)
    colors = (colors + 1.0) * 127.5
    colors = colors.astype('int')
    # draw_keypoints(image, points2D, colors, radius=3)


def normalize_points2D(points2D, height, width):
    """Transform points2D in image coordinates to normalized coordinates.

    # Arguments
        points2D: Numpy array of shape ``(num_keypoints, 2)``.
        height: Int. Height of the image
        width: Int. Width of the image

    # Returns
        Numpy array of shape ``(num_keypoints, 2)``.
    """
    image_shape = np.array([width, height])
    points2D = points2D / image_shape  # [0, W], [0, H] -> [0,  1], [0,  1]
    points2D = 2.0 * points2D          # [0, 1], [0, 1] -> [0,  2], [0,  2]
    points2D = points2D - 1.0          # [0, 2], [0, 2] -> [-1, 1], [-1, 1]
    return points2D


def denormalize_points2D(points2D, height, width):
    image_shape = np.array([width, height])
    points2D = points2D + 1.0          # [-1, 1], [-1, 1] -> [0, 2], [0, 2]
    points2D = points2D / 2.0          # [0 , 2], [0 , 2] -> [0, 1], [0, 1]
    points2D = points2D * image_shape  # [0 , 1], [0 , 1] -> [0, W], [0, H]
    return points2D



def flip_y_axis(points2D):
    x, y = np.split(points2D, 2, axis=1)
    points2D = np.concatenate([x, -y], axis=1)
    return points2D


def denormalize_keypoints2(keypoints, height, width):
    # [-1, 1] -> [-127.5, 127.5] -> [0, 255]
    half_sizes = np.array([width, height]) / 2.0
    return (half_sizes * keypoints) + half_sizes


def translate_points2D(points2D, translation):
    if len(points2D.shape) != 2:
        raise ValueError('Invalid points2D shape')
    if len(translation) != 2:
        raise ValueError('Invalid translation lenght')
    num_keypoints = len(points2D)
    height, width = translation
    x_translation = np.full((num_keypoints, 1), width)
    y_translation = np.full((num_keypoints, 1), height)
    translation = np.concatenate([x_translation, y_translation], axis=1)
    translated_points2D = translation - points2D
    return translated_points2D


def denormalize_keypoints(keypoints, height, width):
    """Transform normalized keypoint coordinates into image coordinates

    # Arguments
        keypoints: Numpy array of shape ``(num_keypoints, 2)``.
        height: Int. Height of the image
        width: Int. Width of the image

    # Returns
        Numpy array of shape ``(num_keypoints, 2)``.
    """
    for keypoint_arg, keypoint in enumerate(keypoints):
        x, y = keypoint[:2]
        # transform key-point coordinates to image coordinates
        x = (min(max(x, -1), 1) * width / 2 + width / 2) - 0.5
        # flip since the image coordinates for y are flipped
        y = height - 0.5 - (min(max(y, -1), 1) * height / 2 + height / 2)
        x, y = int(round(x)), int(round(y))
        keypoints[keypoint_arg][:2] = [x, y]
    return keypoints


def draw_poses6D(image, poses6D, cube_points3D, camera_intrinsics):
    image = image.astype(float)
    for pose6D in poses6D:
        rotation = quaternion_to_rotation_matrix(pose6D.quaternion)
        rotation = np.squeeze(rotation, axis=2)
        cube_points2D = project_to_image(
            rotation, pose6D.translation,
            cube_points3D, camera_intrinsics)
        cube_points2D = cube_points2D.astype(np.int32)
        image = draw_cube(image, cube_points2D)
    image = image.astype('uint8')
    return image
