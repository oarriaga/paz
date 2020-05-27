import cv2
import numpy as np

UPNP = cv2.SOLVEPNP_UPNP


def normalize_keypoints(keypoints, height, width):
    """Transform keypoints in image coordinates to normalized coordinates
        keypoints: Numpy array of shape (num_keypoints, 2)
        height: Int. Height of the image
        width: Int. Width of the image
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
        keypoints: Numpy array of shape (num_keypoints, 2)
        height: Int. Height of the image
        width: Int. Width of the image
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
    """Cascade classifier with detectMultiScale() method for inference.
    # Arguments
        path: String. Path to default openCV XML format.
    """
    return cv2.CascadeClassifier(path)


def solve_PNP(points3D, points2D, camera, solver):
    """Calculates 6D pose from 3D points and 2D keypoints correspondences.
    # Arguments
        points: Numpy array of shape (num_points, 3).
            Model 3D points known in advance.
        keypoints: Numpy array of shape (num_points, 2).
            Predicted 2D keypoints of object
        camera intrinsics: Numpy array of shape (3, 3) calculated from
        the openCV calibrateCamera function
        solver: Flag from e.g openCV.SOLVEPNP_UPNP
        distortion: Numpy array of shape of 5 elements calculated from
        the openCV calibrateCamera function

    # Returns
        A list containing success flag, rotation and translation components
        of the 6D pose.

    # References
        https://docs.opencv.org/2.4/modules/calib3d/doc/calib3d.html
    """
    return cv2.solvePnP(points3D, points2D, camera.intrinsics,
                        camera.distortion, None, None, False, solver)


def project_points3D(points3D, pose6D, camera):
    """Projects 3D points into a specific pose.
    """
    point2D, jacobian = cv2.projectPoints(
        points3D, pose6D.rotation_vector, pose6D.translation,
        camera.intrinsics, camera.distortion)
    return point2D
