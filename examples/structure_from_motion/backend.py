import cv2 
import numpy as np


def detect_SIFT_features(image):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def brute_force_matcher(descriptor1, descriptor2, k=2, ratio_test=True):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k)
    if ratio_test:
        good_matches = []
        for m, n in matches:
            if m.distance < 0.4*n.distance:
                good_matches.append([m])
    return good_matches


def find_homography_RANSAC(points1, points2):
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    return H, mask
    