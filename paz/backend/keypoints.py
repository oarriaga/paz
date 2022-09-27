from warnings import warn

import cv2
import numpy as np

UPNP = cv2.SOLVEPNP_UPNP
LEVENBERG_MARQUARDT = cv2.SOLVEPNP_ITERATIVE


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


def normalize_keypoints2D(points2D, height, width):
    """Transform points2D in image coordinates to normalized coordinates i.e.
        [U, V] -> [-1, 1]. UV have maximum values of [W, H] respectively.

             Image plane

                 width
           (0,0)-------->  (U)
             |
      height |
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


def denormalize_keypoints2D(points2D, height, width):
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


"""
def denormalize_keypoints(keypoints, height, width):
    for keypoint_arg, keypoint in enumerate(keypoints):
        x, y = keypoint[:2]
        # transform key-point coordinates to image coordinates
        x = (min(max(x, -1), 1) * width / 2 + width / 2) - 0.5
        # flip since the image coordinates for y are flipped
        y = height - 0.5 - (min(max(y, -1), 1) * height / 2 + height / 2)
        x, y = int(round(x)), int(round(y))
        keypoints[keypoint_arg][:2] = [x, y]
    return keypoints
"""


def cascade_classifier(path):
    """OpenCV Cascade classifier.

    # Arguments
        path: String. Path to default openCV XML format.

    # Returns
        OpenCV classifier with ``detectMultiScale`` for inference..
    """
    return cv2.CascadeClassifier(path)


def solve_PNP(points3D, points2D, camera, solver):
    """Calculates 6D pose from 3D points and 2D keypoints correspondences.

    # Arguments
        points3D: Numpy array of shape ``(num_points, 3)``.
            3D points known in advance.
        points2D: Numpy array of shape ``(num_points, 2)``.
            Predicted 2D keypoints of object.
        camera: Instance of ''paz.backend.Camera'' containing as properties
            the ''camera_intrinsics'' a Numpy array of shape ''(3, 3)''
            usually calculated from the openCV ''calibrateCamera'' function,
            and the ''distortion'' a Numpy array of shape ''(5)'' in which the
            elements are usually obtained from the openCV
            ''calibrateCamera'' function.
        solver: Flag from e.g openCV.SOLVEPNP_UPNP.
        distortion: Numpy array of shape of 5 elements calculated from
            the openCV calibrateCamera function.

    # Returns
        A list containing success flag, rotation and translation components
        of the 6D pose.
    """
    return cv2.solvePnP(points3D, points2D, camera.intrinsics,
                        camera.distortion, None, None, False, solver)


def project_points3D(points3D, pose6D, camera):
    """Projects 3D points into a specific pose.

    # Arguments
        points3D: Numpy array of shape ``(num_points, 3)``.
        pose6D: An instance of ``paz.abstract.Pose6D``.
        camera: An instance of ``paz.backend.Camera`` object.

    # Returns
        Numpy array of shape ``(num_points, 2)``
    """
    points2D, jacobian = cv2.projectPoints(
        points3D, pose6D.rotation_vector, pose6D.translation,
        camera.intrinsics, camera.distortion)
    # openCV adds a dimension to projection i.e. (num_points, 1, 2)
    points2D = np.squeeze(points2D, axis=1)
    return points2D


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


def translate_points2D_origin(points2D, coordinates):
    """Translates points2D to a different origin

    # Arguments
        points2D: Array (num_points, 2)
        coordinates: Array (4) containing (x_min, y_min, x_max, y_max)

    # Returns
        Translated points2D array (num_points, 2)
    """
    x_min, y_min, x_max, y_max = coordinates
    points2D[:, 0] = points2D[:, 0] + x_min
    points2D[:, 1] = points2D[:, 1] + y_min
    return points2D


def translate_keypoints(keypoints, translation):
    """Translate keypoints.

    # Arguments
        kepoints: Numpy array of shape ``(num_keypoints, 2)``.
        translation: A list of length two indicating the x,y translation values

    # Returns
        Numpy array
    """
    return keypoints + translation


def _preprocess_image_points2D(image_points2D):
    """Preprocessing image points for openCV's PnPRANSAC

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
        Arguments are row args (V) and col args (U). Image points are in UV
            coordinates; thus, we concatenate them in that order
            i.e. [col_args, row_args]
    """
    row_args = row_args.reshape(-1, 1)
    col_args = col_args.reshape(-1, 1)
    image_points2D = np.concatenate([col_args, row_args], axis=1)  # (U, V)
    return image_points2D


def normalize_keypoints(keypoints, height, width):
    """Transform keypoints in image coordinates to normalized coordinates
    # Arguments
        keypoints: Numpy array of shape ``(num_keypoints, 2)``.
        height: Int. Height of the image
        width: Int. Width of the image
    # Returns
        Numpy array of shape ``(num_keypoints, 2)``.
    """
    warn('DEPRECATED please use denomarlize_points2D')
    normalized_keypoints = np.zeros_like(keypoints, dtype=np.float32)
    for keypoint_arg, keypoint in enumerate(keypoints):
        x, y = keypoint[:2]
        # transform key-point coordinates to image coordinates
        x = (((x + 0.5) - (width / 2.0)) / (width / 2))
        y = (((height - 0.5 - y) - (height / 2.0)) / (height / 2))
        normalized_keypoints[keypoint_arg][:2] = [x, y]
    return normalized_keypoints


def denormalize_keypoints(keypoints, height, width):
    """Transform normalized keypoint coordinates into image coordinates
    # Arguments
        keypoints: Numpy array of shape ``(num_keypoints, 2)``.
        height: Int. Height of the image
        width: Int. Width of the image
    # Returns
        Numpy array of shape ``(num_keypoints, 2)``.
    """
    warn('DEPRECATED please use denomarlize_points2D')
    for keypoint_arg, keypoint in enumerate(keypoints):
        x, y = keypoint[:2]
        # transform key-point coordinates to image coordinates
        x = (min(max(x, -1), 1) * width / 2 + width / 2) - 0.5
        # flip since the image coordinates for y are flipped
        y = height - 0.5 - (min(max(y, -1), 1) * height / 2 + height / 2)
        x, y = int(round(x)), int(round(y))
        keypoints[keypoint_arg][:2] = [x, y]
    return keypoints


def rotate_point2D(point2D, rotation_angle):
    """Rotate keypoint.

    # Arguments
        point2D: keypoint [x, y]
        rotation angle: Int. Angle of rotation.

    # Returns
        List of x and y rotated points
    """
    rotation_angle = np.pi * rotation_angle / 180
    sin_n, cos_n = np.sin(rotation_angle), np.cos(rotation_angle)
    x_rotated = (point2D[0] * cos_n) - (point2D[1] * sin_n)
    y_rotated = (point2D[0] * sin_n) + (point2D[1] * cos_n)
    return [x_rotated, y_rotated]


def transform_keypoint(keypoint, transform):
    """ Transform keypoint.

    # Arguments
        keypoint2D: keypoint [x, y]
        transform: Array. Transformation matrix
    """
    keypoint = np.array([keypoint[0], keypoint[1], 1.]).T
    transformed_keypoint = np.dot(transform, keypoint)
    return transformed_keypoint


def add_offset_to_point(keypoint_location, offset=0):
    """ Add offset to keypoint location

    # Arguments
        keypoint_location: keypoint [y, x]
        offset: Float.
    """
    y, x = keypoint_location
    y = y + offset
    x = x + offset
    return y, x


def flip_keypoints_left_right(keypoints, image_size=(32, 32)):
    """Flip the detected 2D keypoints left to right.

    # Arguments
        keypoints: Array
        image_size: list/tuple
        axis: int

    # Returns
        flipped_keypoints: Numpy array
    """
    x_coordinates, y_coordinates = np.split(keypoints, 2, axis=1)
    flipped_x = image_size[0] - x_coordinates
    keypoints = np.concatenate((flipped_x, y_coordinates), axis=1)
    return keypoints


def compute_orientation_vector(keypoints3D, parents):
    """Compute bone orientations from joint coordinates
       (child joint - parent joint). The returned vectors are normalized.
       For the root joint, it will be a zero vector.

    # Arguments
        keypoints3D : Numpy array [num_keypoints, 3]. Joint coordinates.
        parents: Parents of the keypoints from kinematic chain

    # Returns
        Array [num_keypoints, 3]. The unit vectors from each child joint to
        its parent joint. For the root joint, it's are zero vector.
    """
    delta = []
    for joint_arg in range(len(parents)):
        parent = parents[joint_arg]
        if parent is None:
            delta.append(np.zeros(3))
        else:
            delta.append(keypoints3D[joint_arg] - keypoints3D[parent])
    delta = np.stack(delta, 0)
    return delta


def rotate_keypoints3D(rotation_matrix, keypoints):
    """Rotatate the keypoints by using rotation matrix

    # Arguments
        Rotation matrix [N, 3, 3].
        keypoints [N, 3]

    # Returns
        Rotated keypoints [N, 3]
    """
    keypoint_xyz = np.einsum('ijk, ik -> ij', rotation_matrix, keypoints)
    return keypoint_xyz


def flip_along_x_axis(keypoints, axis=0):
    """Flip the keypoints along the x axis.

    # Arguments
        keypoints: Array
        axis: int/list

    # Returns
        Flipped keypoints: Array
    # """
    x, y, z = np.split(keypoints, 3, axis=1)
    keypoints = np.concatenate((-x, y, z), axis=1)
    return keypoints


def uv_to_vu(keypoints):
    """Flips the uv coordinates to vu.

    # Arguments
        keypoints: Array.
    """
    return keypoints[:, ::-1]
