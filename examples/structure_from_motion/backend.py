import cv2
import numpy as np


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
    return keypoints, descriptors


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
    return [query_indices, train_indices]


# def get_match_descriptors(descriptors1, descriptors2, matches):
#     descriptors1_ = np.zeros((len(matches), 128), dtype=np.float32)
#     descriptors2_ = np.zeros((len(matches), 128), dtype=np.float32)
#     for i, match in enumerate(matches):
#         descriptors1_[i, :] = descriptors1[match[0].queryIdx]
#         descriptors2_[i, :] = descriptors2[match[0].trainIdx]
#     return [descriptors1_, descriptors2_]


def find_homography_RANSAC(points1, points2):
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 0.5)
    return H, mask


def find_fundamental_matrix(points1, points2):
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 0.5, 0.99)
    return F, mask


def recover_pose(E, points1, points2, K):
    points, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    return points, R, t, mask


def triangulate_points(P1, P2, points1, points2):
    points4D = cv2.triangulatePoints(P1, P2, points1, points2)
    points3D = (points4D[:3, :]/points4D[3, :]).T
    return points3D


def compute_essential_matrix(fundamental_matrix, camera_intrinsics):
    essential_matrix = camera_intrinsics.T @ (fundamental_matrix @
                                              camera_intrinsics)
    return essential_matrix
