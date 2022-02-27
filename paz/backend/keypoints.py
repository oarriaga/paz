import cv2
import numpy as np

UPNP = cv2.SOLVEPNP_UPNP
LEVENBERG_MARQUARDT = cv2.SOLVEPNP_ITERATIVE


def normalize_keypoints(keypoints, height, width):
    """Transform keypoints in image coordinates to normalized coordinates

    # Arguments
        keypoints: Numpy array of shape ``(num_keypoints, 2)``.
        height: Int. Height of the image
        width: Int. Width of the image

    # Returns
        Numpy array of shape ``(num_keypoints, 2)``.
    """
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
    for keypoint_arg, keypoint in enumerate(keypoints):
        x, y = keypoint[:2]
        # transform key-point coordinates to image coordinates
        x = (min(max(x, -1), 1) * width / 2 + width / 2) - 0.5
        # flip since the image coordinates for y are flipped
        y = height - 0.5 - (min(max(y, -1), 1) * height / 2 + height / 2)
        x, y = int(round(x)), int(round(y))
        keypoints[keypoint_arg][:2] = [x, y]
    return keypoints


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
    point2D, jacobian = cv2.projectPoints(
        points3D, pose6D.rotation_vector, pose6D.translation,
        camera.intrinsics, camera.distortion)
    return point2D


def translate_keypoints(keypoints, translation):
    """Translate keypoints.

    # Arguments
        kepoints: Numpy array of shape ``(num_keypoints, 2)``.
        translation: A list of length two indicating the x,y translation values

    # Returns
        Numpy array
    """
    return keypoints + translation


def rotate_keypoint(point2D, rotation_angle):
    """Rotate keypoint.

    # Arguments
        point2D: keypoint [x, y]
        rotation angle: Int. Angle of rotation.

    # Returns
        List of x and y rotated points
    """
    rotation_angle = np.pi * rotation_angle / 180
    sn, cs = np.sin(rotation_angle), np.cos(rotation_angle)
    x_rotated = (point2D[0] * cs) - (point2D[1] * sn)
    y_rotated = (point2D[0] * sn) + (point2D[1] * cs)
    return [x_rotated, y_rotated]


def transform_keypoint(keypoint, transform):
    """ Transform keypoint.

    # Arguments
        keypoint2D: keypoint [x, y]
        transform: Numpy array. Transformation matrix
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
