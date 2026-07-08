from collections import namedtuple

import cv2
import numpy as np
import jax.numpy as jp


UPNP = cv2.SOLVEPNP_UPNP
LEVENBERG_MARQUARDT = cv2.SOLVEPNP_ITERATIVE
EPNP = cv2.SOLVEPNP_EPNP
MIN_REQUIRED_POINTS = 4

Pose6D = namedtuple("Pose6D", ["rotation_vector", "translation"])


def build_cube_points3D(width, height, depth):
    half_width, half_height, half_depth = width / 2, height / 2, depth / 2
    point1 = [+half_width, -half_height, +half_depth]
    point2 = [+half_width, -half_height, -half_depth]
    point3 = [-half_width, -half_height, -half_depth]
    point4 = [-half_width, -half_height, +half_depth]
    point5 = [+half_width, +half_height, +half_depth]
    point6 = [+half_width, +half_height, -half_depth]
    point7 = [-half_width, +half_height, -half_depth]
    point8 = [-half_width, +half_height, +half_depth]
    points = [point1, point2, point3, point4, point5, point6, point7, point8]
    return jp.array(points)


def normalize_keypoints2D(points2D, height, width):
    image_shape = jp.array([width, height])
    return (2.0 * points2D / image_shape) - 1.0


def denormalize_keypoints2D(points2D, height, width):
    image_shape = jp.array([width, height])
    return (points2D + 1.0) / 2.0 * image_shape


def rotate_point2D(point2D, rotation_angle):
    angle = jp.pi * rotation_angle / 180.0
    sin_angle, cos_angle = jp.sin(angle), jp.cos(angle)
    x = point2D[0] * cos_angle - point2D[1] * sin_angle
    y = point2D[0] * sin_angle + point2D[1] * cos_angle
    return jp.array([x, y])


def transform_keypoint(keypoint, transform):
    point = jp.array([keypoint[0], keypoint[1], 1.0])
    return transform @ point


def flip_keypoints_left_right(keypoints, width):
    x, y = jp.split(keypoints, 2, axis=1)
    return jp.concatenate([width - x, y], axis=1)


def translate_keypoints(keypoints, translation):
    return keypoints + translation


def uv_to_vu(keypoints):
    return keypoints[:, ::-1]


def rotate_keypoints2D(keypoints, angle, center):
    cos_angle, sin_angle = jp.cos(angle), jp.sin(angle)
    rotation = jp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    return (keypoints - center) @ rotation.T + center


def project_points3D(points3D, pose6D, camera):
    args = (pose6D.translation, camera.intrinsics, camera.distortion)
    points2D, _ = cv2.projectPoints(points3D, pose6D.rotation_vector, *args)
    return jp.squeeze(points2D, axis=1)  # openCV shape (num_points, 1, 2)


def solve_PnP(points2D, points3D, camera, solver=LEVENBERG_MARQUARDT):
    points2D = np.array(points2D, np.float64).reshape((len(points3D), 1, 2))
    args = (camera.intrinsics, camera.distortion, None, None, False, solver)
    (_, rotation_vector, translation) = cv2.solvePnP(points3D, points2D, *args)
    return Pose6D(rotation_vector, translation)


def solve_PnP_RANSAC(points2D, points3D, camera, inlier_thresh=5.0,
                     iterations=100):
    if len(points3D) < MIN_REQUIRED_POINTS:
        return None
    points2D = np.array(points2D, np.float64).reshape((len(points3D), 1, 2))
    points3D = np.array(points3D, np.float64)
    args = (camera.intrinsics, camera.distortion, None, None, False,
            iterations, inlier_thresh, 0.99, None, EPNP)
    success, rotation, translation, inliers = cv2.solvePnPRansac(
        points3D, points2D, *args)
    if not success:
        return None
    return Pose6D(rotation, translation)


def rotation_vector_to_matrix(rotation_vector):
    return cv2.Rodrigues(rotation_vector)[0]


def solve_pose_matrix_RANSAC(points2D, points3D, camera, max_points=1500,
                             seed=0):
    if len(points3D) > max_points:
        choice = np.random.RandomState(seed).choice(len(points3D), max_points, False)  # fmt: skip
        points2D, points3D = points2D[choice], points3D[choice]
    pose6D = solve_PnP_RANSAC(points2D, points3D, camera)
    if pose6D is None:
        return None
    rotation = rotation_vector_to_matrix(pose6D.rotation_vector)
    return rotation, np.asarray(pose6D.translation).reshape(3)
