import numpy as np
from paz.backend.image.draw import GREEN
from paz.backend.image import draw_line, draw_dot
import cv2


def build_cube_points3D(width, height, depth):
    """Build the 3D points of a cube in the openCV coordinate system:
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
    """Preprocessing image points for PnPRANSAC

    # Arguments
        image_points2D: Array of shape (num_points, 2)

    # Returns
        Contiguous float64 array of shape (num_points, 1, 2)
    """
    num_points = len(image_points2D)
    image_points2D = image_points2D.reshape(num_points, 1, 2)
    image_points2D = image_points2D.astype(np.float64)
    image_points2D = np.ascontiguousarray(image_points2D)
    return image_points2D


def solve_PnP_RANSAC(object_points3D, image_points2D, camera_intrinsics,
                     inlier_threshold=5, num_iterations=100):
    """Returns rotation (Roc) and translation (Toc) vectors that transform
        3D points in object frame to camera frame.

                               O------------O
                              /|           /|
                             / |          / |
                            O------------O  |
                            |  |    z    |  |
                            |  O____|____|__O
                            |  /    |___y|  /   object
                            | /    /     | /  coordinates
                            |/    x      |/
                            O------------O
                                   ___
                   Z                |
                  /                 | Rco, Tco
                 /_____X     <------|
                 |
                 |    camera
                 Y  coordinates

    # Arguments
        object_points3D: Array (num_points, 3). Points 3D in object reference
            frame. Represented as (0) in image above.
        image_points2D: Array (num_points, 2). Points in 2D in camera UV space.
        camera_intrinsics: Array of shape (3, 3). Diagonal elements represent
            focal lenghts and last column the image center translation.
        inlier_threshold: Number of inliers for RANSAC method.
        num_iterations: Maximum number of iterations.

    # Returns
        Rotation vector in axis-angle form (3) and translation vector (3).
    """
    if ((len(object_points3D) < 4) or (len(image_points2D) < 4)):
        raise ValueError('Solve PnP requires at least 4 3D and 2D points')
    image_points2D = _preprocess_image_points2D(image_points2D)
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points3D, image_points2D, camera_intrinsics, None,
        flags=cv2.SOLVEPNP_EPNP, reprojectionError=inlier_threshold,
        iterationsCount=num_iterations)
    translation = np.squeeze(translation, 1)
    return success, rotation_vector, translation


def project_to_image(rotation, translation, points3D, camera_intrinsics):
    """Project points3D to image plane using a perspective transformation.

              Image plane

           (0,0)-------->  (U)
             |
             |
             |
             v

            (V)

    # Arguments
        rotation: Array (3, 3). Rotation matrix (Rco).
        translation: Array (3). Translation (Tco).
        points3D: Array (num_points, 3). Points 3D in object frame.
        camera_intrinsics: Array of shape (3, 3). Diagonal elements represent
            focal lenghts and last column the image center translation.

    # Returns
        Array (num_points, 2) in UV image space.
    """
    if rotation.shape != (3, 3):
        raise ValueError('Rotation matrix is not of shape (3, 3)')
    if len(translation) != 3:
        raise ValueError('Translation vector is not of length 3')
    if len(points3D.shape) != 2:
        raise ValueError('Points3D should have a shape (num_points, 3)')
    if points3D.shape[1] != 3:
        raise ValueError('Points3D should have a shape (num_points, 3)')
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


def replace_lower_than_threshold(source, threshold=1e-3, replacement=0.0):
    """Replace values from source that are lower than the given threshold.

    # Arguments
        source: Array.
        threshold: Float. Values lower than this value will be replaced.
        replacement: Float. Value taken by elements lower than threshold.

    # Returns
        Array of same shape as source.
    """
    lower_than_epsilon = source < threshold
    source[lower_than_epsilon] = replacement
    return source


def arguments_to_image_points2D(row_args, col_args):
    """Convert array arguments into UV coordinates.

              Image plane

           (0,0)-------->  (U)
             |
             |
             |
             v

            (V)

    # Arguments
        row_args: Array (num_rows).
        col_args: Array (num_cols).

    # Returns
        Array (num_cols, num_rows) representing points2D in UV space.

    # Notes
        Arguments are row args (V) and col args (U). Iamge points are in UV
            coordinates; thus, we concatenate them in that order
            i.e. [col_args, row_args]
    """
    row_args = row_args.reshape(-1, 1)
    col_args = col_args.reshape(-1, 1)
    image_points2D = np.concatenate([col_args, row_args], axis=1)
    return image_points2D


def points3D_to_RGB(points3D, object_sizes):
    """Transforms points3D in object frame to RGB color space.
    # Arguments
        points3D: Array (num_points, 3). Points3D a
        object_sizes: List (3) indicating the
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


# TODO change to processor
def draw_masks(image, points, object_sizes):
    for points2D, points3D in points:
        colors = points3D_to_RGB(points3D, object_sizes)
        image = draw_points2D(image, points2D, colors)
    return image


def draw_points2D(image, points2D, colors):
    """Draws a pixel for all points2D in UV space using only numpy.

    # Arguments
        image: Array (H, W).
        keypoints: Array (num_points, U, V). Keypoints in image space
        colors: Array (num_points, 3). Colors in RGB space.

    # Returns
        Array with drawn points.
    """
    keypoints = points2D.astype(int)
    U = keypoints[:, 0]
    V = keypoints[:, 1]
    image[V, U, :] = colors
    return image


def draw_points2D_(image, keypoints, colors, radius=1):
    for (u, v), (R, G, B) in zip(keypoints, colors):
        color = (int(R), int(G), int(B))
        draw_dot(image, (u, v), color, radius)
    return image


def normalize_points2D(points2D, height, width):
    """Transform points2D in image coordinates to normalized coordinates i.e.
        [U, V] -> [-1, 1]. UV have maximum values of [W, H] respectively.

             Image plane

           (0,0)-------->  (U)
             |
             |
             |
             v

            (V)

    # Arguments
        points2D: Numpy array of shape (num_keypoints, 2).
        height: Int. Height of the image
        width: Int. Width of the image

    # Returns
        Numpy array of shape (num_keypoints, 2).
    """
    image_shape = np.array([width, height])
    points2D = points2D / image_shape  # [W, 0], [0, H] -> [1,  0], [0,  1]
    points2D = 2.0 * points2D          # [1, 0], [0, 1] -> [2,  0], [0,  2]
    points2D = points2D - 1.0          # [2, 0], [0, 2] -> [-1, 1], [-1, 1]
    return points2D


def denormalize_points2D(points2D, height, width):
    """Transform nomralized points2D to image UV coordinates i.e.
        [-1, 1] -> [U, V]. UV have maximum values of [W, H] respectively.

             Image plane

           (0,0)-------->  (U)
             |
             |
             |
             v

            (V)

    # Arguments
        points2D: Numpy array of shape (num_keypoints, 2).
        height: Int. Height of the image
        width: Int. Width of the image

    # Returns
        Numpy array of shape (num_keypoints, 2).
    """
    image_shape = np.array([width, height])
    points2D = points2D + 1.0          # [-1, 1], [-1, 1] -> [2, 0], [0, 2]
    points2D = points2D / 2.0          # [2 , 0], [0 , 2] -> [1, 0], [0, 1]
    points2D = points2D * image_shape  # [1 , 0], [0 , 1] -> [W, 0], [0, H]
    return points2D


def draw_pose6D(image, pose6D, cube_points3D, camera_intrinsics):
    """Draws pose6D by projecting cube3D to image space with camera intrinsics.

    # Arguments
        image: Array (H, W, 3)
        pose6D: paz message Pose6D with quaternion and translation values.
        cube3D: Array (8, 3). Cube 3D points in object frame.
        camera_intrinsics: Array of shape (3, 3). Diagonal elements represent
            focal lenghts and last column the image center translation.

    # Returns
        Original image array (H, W, 3) with drawn cube points.
    """
    quaternion, translation = pose6D.quaternion, pose6D.translation
    rotation = quaternion_to_rotation_matrix(quaternion)
    rotation = np.squeeze(rotation, axis=2)
    cube_points2D = project_to_image(
        rotation, translation, cube_points3D, camera_intrinsics)
    cube_points2D = cube_points2D.astype(np.int32)
    image = draw_cube(image, cube_points2D)
    return image


def draw_poses6D(image, poses6D, cube_points3D, camera_intrinsics):
    """Draws pose6D by projecting cube3D to image space with camera intrinsics.

    # Arguments
        image: Array (H, W, 3)
        pose6D: List paz messages Pose6D with quaternions and translations.
        cube3D: Array (8, 3). Cube 3D points in object frame.
        camera_intrinsics: Array of shape (3, 3). Diagonal elements represent
            focal lenghts and last column the image center translation.

    # Returns
        Original image array (H, W, 3) with drawn cube points for all poses6D.
    """
    for pose6D in poses6D:
        image = draw_pose6D(image, pose6D, cube_points3D, camera_intrinsics)
    return image


def homogenous_quaternion_to_rotation_matrix(quaternion):
    """Transforms quaternion to rotation matrix.

    # Arguments
        quaternion: Array containing quaternion value [q1, q2, q3, w0].

    # Returns
        Rotation matrix [3, 3].

    # Note
        If quaternion is not a unit quaternion the rotation matrix is not
        unitary but still orthogonal i.e. the outputted rotation matrix is
        a scalar multiple of a rotation matrix.
    """
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


def quaternion_to_rotation_matrix(quaternion):
    """Transforms quaternion to rotation matrix.

    # Arguments
        quaternion: Array containing quaternion value [q1, q2, q3, w0].

    # Returns
        Rotation matrix [3, 3].

    # Note
        "If the quaternion "is not a unit quaternion then the homogeneous form
        is still a scalar multiple of a rotation matrix, while the
        inhomogeneous form is in general no longer an orthogonal matrix.
        This is why in numerical work the homogeneous form is to be preferred
        if distortion is to be avoided." [wikipedia](https://en.wikipedia.org/
            wiki/Conversion_between_quaternions_and_Euler_angles)
    """
    matrix = homogenous_quaternion_to_rotation_matrix(quaternion)
    return matrix


def rotation_vector_to_rotation_matrix(rotation_vector):
    """Transforms rotation vector (axis-angle) form to rotation matrix.

    # Arguments
        rotation_vector: Array (3). Rotation vector in axis-angle form.

    # Returns
        Array (3, 3) rotation matrix.
    """
    rotation_matrix = np.eye(3)
    cv2.Rodrigues(rotation_vector, rotation_matrix)
    return rotation_matrix


def to_affine_matrix(rotation_matrix, translation):
    """Builds affine matrix from rotation matrix and translation vector.

    # Arguments
        rotation_matrix: Array (3, 3). Representing a rotation matrix.
        translation: Array (3). Translation vector.

    # Returns
        Array (4, 4) representing an affine matrix.
    """
    if len(translation) != 3:
        raise ValueError('Translation should be of lenght 3')
    if rotation_matrix.shape != (3, 3):
        raise ValueError('Rotation matrix should be of shape (3, 3)')
    translation = translation.reshape(3, 1)
    affine_top = np.concatenate([rotation_matrix, translation], axis=1)
    affine_row = np.array([[0.0, 0.0, 0.0, 1.0]])
    affine_matrix = np.concatenate([affine_top, affine_row], axis=0)
    return affine_matrix


def image_to_normalized_device_coordinates(image):
    """Map image value from [0, 255] -> [-1, 1].
    """
    return (image / 127.5) - 1.0


def normalized_device_coordinates_to_image(image):
    """Map normalized value from [-1, 1] -> [0, 255].
    """
    return (image + 1.0) * 127.5


def build_rotation_matrix_z(angle):
    """Builds rotation matrix in Z axis.
    # Arguments
        angle: Float. Angle in radians.
    # Return
        Array (3, 3) rotation matrix in Z axis.
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix_z = np.array([[+cos_angle, -sin_angle, 0.0],
                                  [+sin_angle, +cos_angle, 0.0],
                                  [0.0, 0.0, 1.0]])
    return rotation_matrix_z


def rotate_image(image, rotation_matrix):
    """Rotates an image with a symmetry.
    # Arguments
        image: Array (H, W, 3) with domain [0, 255].
        rotation_matrix: Array (3, 3).

    # Returns
        Array (H, W, 3) with domain [0, 255]
    """
    mask_image = np.sum(image, axis=-1, keepdims=True) != 0
    image = image_to_normalized_device_coordinates(image)
    rotated_image = np.einsum('ij,klj->kli', rotation_matrix, image)
    rotated_image = normalized_device_coordinates_to_image(rotated_image)
    rotated_image = np.clip(rotated_image, a_min=0.0, a_max=255.0)
    rotated_image = rotated_image * mask_image
    return rotated_image
