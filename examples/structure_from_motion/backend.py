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


def find_fundamental_matrix(points1, points2):
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    return F, mask


def recover_pose(E, points1, points2, K):
    points, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    return points, R, t, mask


def triangulate_points(P1, P2, points1, points2):
    points4D = cv2.triangulatePoints(P1, P2, points1, points2)
    points3D = (points4D[:3, :]/points4D[3, :]).T
    return points3D


def solve_PNP(points3D, points2D, projection_matrix):
    points, R, t = cv2.solvePnPRansac(points3D, points2D, projection_matrix)
    return points, R, t
