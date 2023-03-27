import cv2
import numpy as np
from skimage.measure import ransac


def detect_ORB_fratures(image):
    orb = cv2.ORB_create()
    # detection
    points = cv2.goodFeaturesToTrack(np.mean(image, axis=2).astype(np.uint8),
                                     3000, qualityLevel=0.01, minDistance=7)

    # extraction
    keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in points]
    keypoints, descriptors = orb.compute(image, keypoints)
    return keypoints, descriptors


def detect_SIFT_features(image):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return np.array(keypoints), np.array(descriptors)


def brute_force_matcher(descriptor1, descriptor2, k=2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k)
    return matches


def match_ratio_test(matches, ratio=0.75):
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])
    return good_matches


def get_match_points(keypoints1, keypoints2, matches):
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match[0].queryIdx].pt
        points2[i, :] = keypoints2[match[0].trainIdx].pt
    return [points1, points2]


def get_match_indices(matches):
    query_indices = []
    train_indices = []
    for i, match in enumerate(matches):
        query_indices.append(match[0].queryIdx)
        train_indices.append(match[0].trainIdx)
    return [np.array(query_indices), np.array(train_indices)]


def find_homography_RANSAC(points1, points2):
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 0.5)
    return H, mask


def find_fundamental_matrix(points1, points2):
    F, mask = cv2.findFundamentalMat(points1, points2,
                                     cv2.FM_RANSAC, 0.5, 0.99)
    return F, mask


def recover_pose(E, points1, points2, K):
    points, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    return R, t


def triangulate_points(P1, P2, points1, points2):
    points4D = cv2.triangulatePoints(P1, P2, points1, points2)
    points3D = (points4D[:3, :]/points4D[3, :]).T
    return points3D


def compute_essential_matrix(fundamental_matrix, camera_intrinsics):
    essential_matrix = camera_intrinsics.T @ (fundamental_matrix @
                                              camera_intrinsics)
    return essential_matrix


def skimgage_ransac(points1, points2, model_class, min_samples=8,
                    residual_thresh=0.5, max_iterations=1000):
    model, inliers = ransac((points1, points2), model_class, min_samples,
                            residual_thresh, max_trials=max_iterations)
    return model, inliers


def contruct_projection_matrix(rotation, translation):
    projection_matrix = rotation @ np.eye(3, 4)
    projection_matrix[:3, 3] = translation.ravel()
    return projection_matrix


def custom_ransac(points1, points2, min_samples=8, residual_threshold=2,
                  iterations=1000):
    best_inliers = []
    best_homography = None
    best_distances_sum = np.inf
    for arg in range(iterations):
        # Randomly select four matches
        indices = np.random.choice(len(points1), min_samples, replace=False)
        src_pts = np.float32([points1[idx] for idx in indices]).reshape(-1, 2)
        dst_pts = np.float32([points2[idx] for idx in indices]).reshape(-1, 2)
        # Compute homography using these matches
        homography, _ = cv2.findHomography(src_pts, dst_pts,
                                           cv2.RANSAC, residual_threshold)
        # Find inliers using the computed homography
        dst_pts_pred = cv2.perspectiveTransform(
            points1.reshape(-1, 1, 2), homography).reshape(-1, 2)
        distances = np.linalg.norm(points2 - dst_pts_pred, axis=1)
        distances_sum = distances.dot(distances)
        inliers = np.where(distances < residual_threshold)[0]

        # If this iteration has more inliers than the previous best, update
        if (len(inliers) > len(best_inliers) or
                (len(inliers) == len(best_inliers) and
                    distances_sum < best_distances_sum)):
            best_inliers = inliers
            best_homography = homography
            best_distances_sum = distances_sum

    return best_homography, best_inliers
