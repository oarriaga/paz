import cv2
import numpy as np
from scipy.spatial import cKDTree
import math


def detect_ORB_fratures(image):
    """
    Detect ORB features in the image.

    # Arguments
        image -- numpy array of shape (height, width, 3)
                containing the input image

    # Returns
        keypoints -- list of KeyPoint objects
                    containing the keypoints of the image
    """
    orb = cv2.ORB_create()
    # detection
    points = cv2.goodFeaturesToTrack(np.mean(image, axis=2).astype(np.uint8),
                                     3000, qualityLevel=0.01, minDistance=7)

    # extraction
    keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in points]
    keypoints, descriptors = orb.compute(image, keypoints)
    return keypoints, descriptors


def detect_SIFT_features(image):
    """
    Detect SIFT features in the image.

    # Arguments
        image -- numpy array of shape (height, width, 3)
                containing the input image

    # Returns
        keypoints -- numpy array of shape (n_keypoints,)
                    containing the keypoints of the image
        descriptors -- numpy array of shape (n_keypoints, 128)
                    containing the descriptors of the image
    """
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return np.array(keypoints), np.array(descriptors)


def brute_force_matcher(descriptor1, descriptor2, k=2):
    """
    Perform the brute force matching between the descriptors of two images.

    # Arguments
        descriptor1 -- numpy array of shape (n_descriptor1, 2)
                    containing the descriptors of the first image
        descriptor2 -- numpy array of shape (n_descriptor2, 128)
                    containing the descriptors of the second image
        k -- int
            number of nearest neighbors to return

    # Returns
        matches -- list of lists of DMatch objects
                containing the matches between the descriptor of two images
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k)
    return matches


def FLANN_matcher(descriptor1, descriptor2, k):
    """
    Perform the FLANN matching between the descriptors of two images.

    # Arguments
        descriptor1 -- numpy array of shape (n_descriptor1, 2)
                       containing the descriptors of the first image
        descriptor2 -- numpy array of shape (n_descriptor2, 128)
                       containing the descriptors of the second image
        k -- int
             number of nearest neighbors to return

    # Returns
        matches -- list of lists of DMatch objects
                containing the matches between the descriptor of two images
    """
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptor1, descriptor2, k)
    return matches


def match_ratio_test(matches, ratio=0.75):
    """
    Perform the ratio test to filter the matches.

    # Arguments
        matches -- list of lists of DMatch objects
                containing the matches between the keypoints of two images
        ratio -- float
                 ratio between the distance of the best match and the second
                 best match

    # Returns
        good_matches -- list of lists of DMatch objects
                        containing the matches between the keypoints of two
                        images after the ratio test
    """
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])
    return good_matches


def get_match_points(keypoints1, keypoints2, matches):
    """
    Get the points corresponding to the matches.

    # Arguments
        keypoints1 -- list of KeyPoint objects
                    containing the keypoints of the first image
        keypoints2 -- list of KeyPoint objects
                    containing the keypoints of the second image
        matches -- list of lists of DMatch objects
                containing the matches between the keypoints of two images

    # Returns
        points1 -- numpy array of shape (n_matches, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_matches, 2)
                containing the 2D points in the second image
    """
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match[0].queryIdx].pt
        points2[i, :] = keypoints2[match[0].trainIdx].pt
    return [points1, points2]


def get_match_indices(matches):
    """
    Get the indices of the matches.

    # Arguments
        matches -- list of lists of DMatch objects
                containing the matches between the keypoints of two images

    # Returns
        query_indices -- numpy array of shape (n_matches, 1)
                        containing the indices of the keypoints in the
                        first image
        train_indices -- numpy array of shape (n_matches, 1)
                        containing the indices of the
                        keypoints in the second image
    """
    query_indices = []
    train_indices = []
    for i, match in enumerate(matches):
        query_indices.append(match[0].queryIdx)
        train_indices.append(match[0].trainIdx)
    return [np.array(query_indices), np.array(train_indices)]


def find_homography_RANSAC_cv(points1, points2, ransacReprojThreshold=0.5,
                              maxIters=1000):
    """
    Compute the homography matrix from corresponding points using the RANSAC
    algorithm.

    # Arguments
        points1 -- numpy array of shape (n_points, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_points, 2)
                containing the 2D points in the second image

    # Returns
        H -- numpy array of shape (3, 3)
            containing the homography matrix
        mask -- numpy array of shape (n_points, 1)
                containing the mask for the inliers
    """
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC,
                                 ransacReprojThreshold, maxIters=maxIters)
    return H, mask


def find_fundamental_matrix_cv(points1, points2, ransacReprojThreshold=0.5,
                               confidence=0.99, maxIters=1000):
    """
    Compute the fundamental matrix from corresponding points using the 8-point
    algorithm.

    # Arguments
        points1 -- numpy array of shape (n_points, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_points, 2)
                containing the 2D points in the second image

    # Returns
        F -- numpy array of shape (3, 3)
            containing the fundamental matrix
        mask -- numpy array of shape (n_points, 1)
                containing the mask for the inliers
    """
    F, mask = cv2.findFundamentalMat(
        points1, points2, cv2.FM_RANSAC, ransacReprojThreshold, confidence,
        maxIters=maxIters)
    return F, mask


def compute_essential_matrix(fundamental_matrix, camera_intrinsics):
    """
    Compute the essential matrix from a fundamental matrix and camera
    intrinsics.

    # Arguments
        fundamental_matrix -- numpy array of shape (3, 3)
                            containing the fundamental matrix
        camera_intrinsics -- numpy array of shape (3, 3)
                            containing the camera intrinsics

    # Returns
        essential_matrix -- numpy array of shape (3, 3)
                            containing the essential matrix
    """
    E = np.matmul(np.matmul(camera_intrinsics.T, fundamental_matrix),
                  camera_intrinsics)

    # Ensure rank-2 by performing SVD and zeroing out smallest singular value
    U, S, Vt = np.linalg.svd(E)
    S[2] = 0
    essential_matrix = np.matmul(np.matmul(U, np.diag(S)), Vt)
    return essential_matrix


def recover_pose_cv(E, points1, points2, K):
    """
    Recover the pose of a second camera from the essential matrix.

    # Arguments
        E -- numpy array of shape (3, 3)
            containing the essential matrix
        points1 -- numpy array of shape (n_points, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_points, 2)
                containing the 2D points in the second image
        K -- numpy array of shape (3, 3)
            containing the intrinsic camera matrix

    # Returns
        R -- numpy array of shape (3, 3)
            containing the rotation matrix
        t -- numpy array of shape (3, 1)
             containing the translation vector
        """
    points, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    return R, t


def triangulate_points_cv(P1, P2, points1, points2):
    """
    Triangulate a set of points from two cameras.

    # Arguments
        P1 -- numpy array of shape (3, 4)
            containing the projection matrix of the first camera
        P2 -- numpy array of shape (3, 4)
            containing the projection matrix of the second camera
        points1 -- numpy array of shape (n_points, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_points, 2)
                containing the 2D points in the second image

    # Returns
        points3D -- numpy array of shape (n_points, 3)
                    containing the triangulated 3D points
    """
    points4D = cv2.triangulatePoints(P1, P2, points1, points2)
    points3D = (points4D[:3, :]/points4D[3, :]).T
    return points3D


def contruct_projection_matrix(rotation, translation):
    """
    Construct a projection matrix from a rotation and translation.

    # Arguments
        rotation -- numpy array of shape (3, 3)
                    containing the rotation matrix
        translation -- numpy array of shape (3, 1)
                       containing the translation vector

    # Returns
        projection_matrix -- numpy array of shape (3, 4)
                            containing the projection matrix
    """
    projection_matrix = rotation @ np.eye(3, 4)
    projection_matrix[:3, 3] = translation.ravel()
    return projection_matrix


def center_and_normalize_points(points):
    """
    Center and normalize a set of 2D points.

    # Arguments
        points -- numpy array of shape (n_points, 2)
                  containing the 2D points

    # Returns
        T -- numpy array of shape (3, 3)
             normalization matrix that was applied to the points
        normalized_points -- numpy array of shape (n_points, 2)
                            containing the normalized points
    """
    N, D = points.shape

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    rms_distance = np.sqrt(np.sum(centered_points ** 2) / N)
    scale = np.sqrt(D) / rms_distance

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    homogeneous_points = np.concatenate([points,
                                         np.ones((len(points), 1))], axis=1)
    normalized_points = np.dot(T, homogeneous_points.T).T[:, :2]

    return T, normalized_points


def compute_fundamental_matrix_np(points1, points2):
    """
    Compute the fundamental matrix from corresponding points using
    the eight-point algorithm.

    Algorithm:
    0. Center and normalize points
    1. Construct the M x 9 matrix A
    2. Find the SVD of ATA
    3. Entries of F are the elements of column of
    V corresponding to the least singular value
    4. (Enforce rank 2 constraint on F)
    5. (Un-normalize F)

    # Arguments
        points1 -- numpy array of shape (n_points, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_points, 2)
                containing the 2D points in the second image

    # Returns
        F -- numpy array of shape (3, 3)
            containing the fundamental matrix
    """
    N, D = points1.shape

    T1, points1 = center_and_normalize_points(points1)
    T2, points2 = center_and_normalize_points(points2)

    values = []
    for i in range(len(points1)):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        values.append([x1 * x2, x2 * y1, x2, y2 * x1, y1 * y2, y2, x1, y1, 1])
    A = np.vstack(values)

    U, S, V = np.linalg.svd(A)

    # The fundamental matrix is the last column of V
    F = V[-1, :].reshape((3, 3))

    # Enforce the rank-2 constraint on F
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V

    # De-normalize F
    F = T2.T @ F @ T1
    return F


def compute_sampson_distance(F, points1, points2):
    """
    Compute the Sampson distance between two sets of corresponding points,
    given the fundamental matrix F.

    # Arguments
        F -- numpy array of shape (3, 3)
            representing the fundamental matrix
        points1 -- numpy array of shape (n_points, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_points, 2)
                containing the corresponding 2D points in the second image

    # Returns
        distance -- numpy array of shape (n_points,)
                    containing the Sampson distance for each
                    corresponding pair of points
    """
    ones = np.ones((points1.shape[0], 1))
    points1_homogeneous = np.hstack((points1, ones))
    points2_homogeneous = np.hstack((points2, ones))

    points1_transformed = np.dot(F, points1_homogeneous.T)
    points2_transformed = np.dot(F.T, points2_homogeneous.T)

    numerator = np.sum(points2_homogeneous * points1_transformed.T, axis=1)
    sum_points1 = np.sum(points1_transformed[:2, :] ** 2, axis=0)
    sum_points2 = np.sum(points2_transformed[:2, :] ** 2, axis=0)
    denominator = sum_points1 + sum_points2

    distance = np.abs(numerator) / np.sqrt(denominator)
    return distance


def estimate_fundamental_matrix_ransac_np(points1, points2, min_samples=8,
                                          residual_threshold=0.5,
                                          max_trials=1000):
    """
    Estimate the fundamental matrix between two sets of corresponding
    points using RANSAC.

    # Arguments
        points1 -- numpy array of shape (n_points, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_points, 2)
                containing the corresponding 2D points in the second image
        min_samples -- int
                    minimum number of samples required to fit the model
        residual_threshold -- float
                            threshold for considering a point an inlier
        max_trials -- int
                    maximum number of iterations to run RANSAC

    # Returns
        F -- numpy array of shape (3, 3)
            representing the estimated fundamental matrix
        inliers -- boolean numpy array of shape (n_points,)
                indicating which points are inliers to the
                estimated fundamental matrix
    """
    best_F = None
    best_inliers = None
    best_num_inliers = 0
    best_distance_sum = np.inf

    for arg in range(max_trials):
        indices = np.random.choice(len(points1), min_samples, replace=False)
        sample_points1 = points1[indices]
        sample_points2 = points2[indices]

        F = compute_fundamental_matrix_np(sample_points1, sample_points2)
        distance = np.abs(compute_sampson_distance(F, points1, points2))

        inliers = distance < residual_threshold
        num_inliers = np.count_nonzero(inliers)
        distance_sum = distance.dot(distance)

        if (num_inliers > best_num_inliers or
                (num_inliers == best_num_inliers and
                    distance_sum < best_distance_sum)):
            best_F = F
            best_inliers = inliers
            best_num_inliers = num_inliers
            best_distance_sum = distance_sum

    if best_F is None:
        print("No inliers found. Model not fitted")
    return best_F, best_inliers


def estimate_homography_ransac_np(points1, points2, min_samples=8,
                                  residual_threshold=2, max_trials=1000):
    """
    Estimate the homography between two sets of corresponding points
    using RANSAC.

    # Arguments
        points1 -- numpy array of shape (n_points, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_points, 2)
                containing the corresponding 2D points in the second image
        min_samples -- int
                    minimum number of samples required to fit the model
        residual_threshold -- float
                            threshold for considering a point an inlier
        max_trials -- int
                    maximum number of iterations to run RANSAC

    # Returns
        H -- numpy array of shape (3, 3)
            representing the estimated homography
        inliers -- boolean numpy array of shape (n_points,)
                indicating which points are inliers to the
    """
    best_H = None
    best_inliers = None
    best_num_inliers = 0
    best_distance_sum = np.inf
    for arg in range(max_trials):
        indices = np.random.choice(len(points1), min_samples, replace=False)
        sample_points1 = points1[indices]
        sample_points2 = points2[indices]
        H, _ = cv2.findHomography(sample_points1, sample_points2,
                                  cv2.RANSAC, residual_threshold)
        points2_predected = cv2.perspectiveTransform(
            points1.reshape(-1, 1, 2), H).reshape(-1, 2)

        distance = np.linalg.norm(points2 - points2_predected,
                                  axis=1)
        inliers = distance < residual_threshold

        distance_sum = distance.dot(distance)
        num_inliers = np.count_nonzero(inliers)

        if (num_inliers > best_num_inliers or
                (num_inliers == best_num_inliers and
                    distance_sum < best_distance_sum)):
            best_H = H
            best_inliers = inliers
            best_num_inliers = num_inliers
            best_distance_sum = distance_sum

    return best_H, best_inliers


def triangulate_points_np(P1, P2, points1, points2):
    """
    Triangulate a set of corresponding points using the linear method.

    # Arguments
        P1 -- numpy array of shape (3, 4)
            representing the projection matrix of the first camera
        P2 -- numpy array of shape (3, 4)
            representing the projection matrix of the second camera
        points1 -- numpy array of shape (n_points, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_points, 2)
                containing the corresponding 2D points in the second image

    # Returns
        points3D -- numpy array of shape (n_points, 3)
                    containing the triangulated 3D points
    """
    ones = np.ones((points1.shape[0], 1))
    points1 = np.hstack((points1, ones))
    points2 = np.hstack((points2, ones))

    points3D = np.zeros((points1.shape[0], 3))
    for i in range(points1.shape[0]):
        A = np.zeros((4, 4))
        A[0, :] = points1[i, 0] * P1[2, :] - P1[0, :]
        A[1, :] = points1[i, 1] * P1[2, :] - P1[1, :]
        A[2, :] = points2[i, 0] * P2[2, :] - P2[0, :]
        A[3, :] = points2[i, 1] * P2[2, :] - P2[1, :]

        _, _, V = np.linalg.svd(A)
        points3D[i, :] = V[-1, :3] / V[-1, 3]
    return points3D


def compute_recover_pose_np(E, points1, points2, K):
    """
    Recover the pose of the second camera from the essential matrix.
    Steps:
        1. Estimating the 4 possible solutions
        2. 3D triangulation of all inlier points
        3. Cheirality check: Basically checking if the triangulated point is
           in front of both cameras (positive z coordinate)
        4. Choose a solutions and flag all inliers that fail Cheirality
           check as outliers.

    # Arguments
        E -- numpy array of shape (3, 3)
            representing the essential matrix
        points1 -- numpy array of shape (n_points, 2)
                containing the 2D points in the first image
        points2 -- numpy array of shape (n_points, 2)
                containing the corresponding 2D points in the second image
        K -- numpy array of shape (3, 3)
            representing the intrinsic camera matrix

    # Returns
        R -- numpy array of shape (3, 3)
            representing the rotation matrix
        t -- numpy array of shape (3,)
            representing the translation vector
        inliers -- boolean numpy array of shape (n_points,)
                indicating which points are inliers to the
                estimated essential matrix
    """

    U, D, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ V
    R2 = U @ W.T @ V
    t1 = U[:, 2]
    t2 = -U[:, 2]

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2s = [K @ np.hstack((R1, t1.reshape(-1, 1))),
           K @ np.hstack((R1, t2.reshape(-1, 1))),
           K @ np.hstack((R2, t1.reshape(-1, 1))),
           K @ np.hstack((R2, t2.reshape(-1, 1)))]

    best_P2 = None
    best_inliers = None
    best_num_inliers = 0
    for P2 in P2s:
        points3D = triangulate_points_np(P1, P2, points1, points2)
        inliers = points3D[:, 2] > 0
        num_inliers = np.count_nonzero(inliers)
        if num_inliers > best_num_inliers:
            best_P2 = P2
            best_inliers = inliers
            best_num_inliers = num_inliers

    R = best_P2[:, :3]
    t = best_P2[:, 3]
    return R, t, best_inliers


def camera_intrinsics_from_hfov(hfov, H, W):
    """
    Compute the camera matrix from the horizontal field of view.

    # Arguments
        fov -- float
               field of view in degrees
        image -- numpy array of shape (height, width, 3)
                 representing the image

    # Returns
        K -- numpy array of shape (3, 3)
            representing the intrinsic camera matrix
    """
    f = W / (2 * np.tan(np.deg2rad(hfov / 2)))
    K = np.array([[f, 0, W / 2],
                  [0, f, H / 2],
                  [0, 0, 1]])
    return K


def camera_intrinsics_from_dfov(dfov, H, W):
    dfov = np.deg2rad(dfov)
    aspect_ratio = W / H
    hfov = 2 * np.arctan(np.tan(dfov / 2) * aspect_ratio)
    f = W / (2 * np.tan(hfov / 2))
    K = np.array([[f, 0, W/2],
                  [0, f, H/2],
                  [0, 0, 1]])
    return K


def remove_outliers(points, threshold=10):
    """
    Remove outliers from a set of points.

    # Arguments
        points -- numpy array of shape (n_points, d)
                    containing the points
        threshold -- float
                     threshold for outlier removal

    # Returns
        points -- numpy array of shape (n_points, 3)
                    containing the 3D points after outlier removal
    """
    mean = np.mean(points, axis=0)
    distance = np.linalg.norm(points - mean, axis=1)
    inliers = distance < threshold
    return points[inliers], inliers


def remove_outliers_cKDTree(points, k=4, std_dev_threshold=0.1):
    """
    Removes outliers from a set of points.

    # Arguments
        points -- numpy array of shape (n, d) representing n 3D points
        k -- int,
             number of neighbors to use for outlier detection
             std_dev_threshold: float, the number of standard deviations
             from the mean distance at which a point is considered an outlier

    # Returns
        points -- numpy array of shape (m, d)
                  representing the m inlier points
    """

    # Build a KD-Tree for efficient neighbor search
    tree = cKDTree(points)

    # Compute distances to each point's k nearest neighbors
    dists, _ = tree.query(points, k=k+1)

    # Compute the mean distance for each point
    mean_dists = np.mean(dists[:, 1:], axis=1)

    # Compute the standard deviation of the mean distances
    std_dev = np.std(mean_dists)

    # Identify points whose mean distance is more than std_dev_threshold
    # standard deviations from the mean distance
    inliers = np.where(mean_dists < (
        np.mean(mean_dists) + std_dev_threshold * std_dev))[0]

    # Remove outliers and return inliers
    return points[inliers], inliers


def extract_keypoints_RGB(image, keypoints):
    """
    Extract the RGB values of the keypoints.

    # Arguments
        keypoints -- list of cv2.KeyPoint objects
                     representing the keypoints
    # Returns
        colors -- numpy array of shape (n_points, 3)
                  representing the RGB values of the keypoints
    """
    colors = []
    for arg, keypoint in enumerate(keypoints):
        color = image[int(keypoint[1]), int(keypoint[0])]
        colors.append(color)
    return colors
