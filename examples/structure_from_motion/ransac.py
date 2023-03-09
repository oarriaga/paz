import numpy as np
from backend import find_homography_RANSAC


def random_point(points1, points2, k=10):
    index = np.random.choice(range(len(points1)), k)
    print(index)
    print('*******************')
    points_1 = []
    points_2 = []
    for arg in index:
        points_1.append(points1[arg])
        points_2.append(points2[arg])
    return np.array(points_1), np.array(points_2)


def get_error(points, H):
    points1, points2 = points
    num_points = len(points1)
    all_p1 = np.concatenate((points1, np.ones((num_points, 1))), axis=1)
    all_p1 = np.asarray(all_p1)
    estimate_p2 = np.zeros((num_points, 2))
    H = np.asarray(H)
    for i in range(num_points):
        temp = np.asarray(all_p1[i])
        estimate_p2[i] = np.dot(H, temp)[0:2]
    # Compute error
    errors = np.linalg.norm(points2 - estimate_p2, axis=1) ** 2
    return errors


def ransac(matches, threshold, iterations):
    points1, points2 = matches
    num_best_inliers = 0
    best_inliers = []
    best_H = []
    for arg in range(iterations):
        points_1, points_2 = random_point(points1, points2)
        H, _ = find_homography_RANSAC(points_1, points_2)

        #  avoid dividing by zero
        # if np.linalg.matrix_rank(H) < 3:
        #     continue

        errors = get_error([points1, points2], H)
        print('errors', errors)
        idx = np.where(errors < threshold)[0]
        matches = np.asarray(matches)
        inliers = matches[idx]
        print(inliers)

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()

    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H
