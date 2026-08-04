import cv2
import numpy as np
from scipy.optimize import least_squares
import paz


def detect_SIFT_features(image):
    """Detects SIFT keypoints. No JAX equivalent exists, so this stays cv2."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return np.array(keypoints), descriptors


def match_features(descriptors1, descriptors2, ratio=0.75):
    matcher = cv2.FlannBasedMatcher()
    descriptors1 = np.asarray(descriptors1, np.float32)
    descriptors2 = np.asarray(descriptors2, np.float32)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    return [best for best, second in knn_matches
            if best.distance < ratio * second.distance]


def get_match_points(keypoints1, keypoints2, matches):
    points1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
    points2 = np.array([keypoints2[match.trainIdx].pt for match in matches])
    return points1, points2


def get_match_indices(matches):
    query_indices = np.array([match.queryIdx for match in matches])
    train_indices = np.array([match.trainIdx for match in matches])
    return query_indices, train_indices


def solve_PnP_RANSAC(points3D, points2D, camera_intrinsics, inlier_thresh=5):
    points2D = np.array(points2D, np.float64).reshape(-1, 1, 2)
    points3D = np.array(points3D, np.float64)
    args = (points3D, points2D, camera_intrinsics, None)
    _, rotation_vector, translation, _ = cv2.solvePnPRansac(
        *args, reprojectionError=inlier_thresh)
    return rotation_vector.ravel(), translation.ravel()


def refine_camera_pose(rotation, translation, points3D, points2D,
                       camera_intrinsics):
    """Refines a camera pose from reprojection error against fixed points3D.

    Only the 6 pose parameters are optimized (not the point positions): with
    scipy's default dense finite-difference Jacobian, adding one 3D point's
    coordinates as free parameters costs 3 extra residual evaluations per
    solver step, which becomes impractically slow at the hundreds of points
    a real correspondence set has. Refining the pose, then re-triangulating
    with it, still improves the point cloud without that cost.
    """
    axis_angle = paz.angles.rotation_matrix_to_compact_axis_angle(rotation)
    pose = np.concatenate([axis_angle, translation.reshape(-1)])
    args = (points3D, points2D, camera_intrinsics)
    result = least_squares(compute_reprojection_residuals, pose, args=args)
    return cv2.Rodrigues(result.x[:3])[0], result.x[3:6]


def compute_reprojection_residuals(pose, points3D, points2D, camera_intrinsics):
    rotation = cv2.Rodrigues(pose[:3])[0]
    projected = paz.poses.project_to_image(
        rotation, pose[3:6], points3D, camera_intrinsics)
    return np.linalg.norm(points2D - projected, axis=1)


def extract_keypoints_RGB(image, points):
    return np.array([image[int(y), int(x)] for x, y in points])


def remove_outliers(points, threshold=10):
    mean = np.mean(points, axis=0)
    distances = np.linalg.norm(points - mean, axis=1)
    inliers = distances < threshold
    return points[inliers], inliers
