import cv2
import jax.numpy as jp
import numpy as np

import paz


def detect_SIFT(image):
    """Detect SIFT features in the image."""
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return np.array(keypoints), np.array(descriptors)


def match_FLANN(descriptor1, descriptor2, k):
    """Perform the FLANN matching between the descriptors of two images."""
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptor1, descriptor2, k)
    return matches


def test_match_ratio(matches, ratio=0.75):
    good_matches = []
    for m, n in matches:
        if m.distance < (ratio * n.distance):
            good_matches.append([m])
    return good_matches


def get_match_points(keypoints_1, keypoints_2, matches):
    """Get the points corresponding to the matches."""
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints_1[match[0].queryIdx].pt
        points2[i, :] = keypoints_2[match[0].trainIdx].pt
    return [points1, points2]


def compute_sampson_distance(fundamental_matrix, points1, points2):
    """Compute the Sampson distance between two sets of corresponding points."""
    points1_homogeneous = paz.points2D.add_ones(points1)
    points2_homogeneous = paz.points2D.add_ones(points2)
    points1_transformed = jp.dot(fundamental_matrix, points1_homogeneous.T)
    points2_transformed = jp.dot(fundamental_matrix.T, points2_homogeneous.T)
    numerator = jp.sum(points2_homogeneous * points1_transformed.T, axis=1)
    sum_points1 = jp.sum(points1_transformed[:2, :] ** 2, axis=0)
    sum_points2 = jp.sum(points2_transformed[:2, :] ** 2, axis=0)
    denominator = sum_points1 + sum_points2
    distance = jp.abs(numerator) / jp.sqrt(denominator)
    return distance
