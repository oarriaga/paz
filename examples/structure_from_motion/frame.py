import os
import cv2
import numpy as np
from skimage.measure import ransac
np.set_printoptions(suppress=True)
from backend import match_ratio_test, get_match_points, get_match_indices


def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret


def fundamentalToRt(F):
    W = np.mat([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]

    # TODO: Resolve ambiguities in better ways. This is wrong.
    if t[2] < 0:
        t *= -1

    # TODO: UGLY!
    if os.getenv("REVERSE") is not None:
        t *= -1
    return np.linalg.inv(poseRt(R, t))


class EssentialMatrixTransform(object):
    def __init__(self):
        self.params = np.eye(3)

    def __call__(self, coords):
        coords_homogeneous = np.column_stack([coords,
                                              np.ones(coords.shape[0])])
        return coords_homogeneous @ self.params.T

    def estimate(self, src, dst):
        assert src.shape == dst.shape
        assert src.shape[0] >= 8

        # Setup homogeneous linear equation as dst' * F * src = 0.
        A = np.ones((src.shape[0], 9))
        A[:, :2] = src
        A[:, :3] *= dst[:, 0, np.newaxis]
        A[:, 3:5] = src
        A[:, 3:6] *= dst[:, 1, np.newaxis]
        A[:, 6:8] = src

        # Solve for the nullspace of the constraint matrix.
        _, _, V = np.linalg.svd(A)
        F = V[-1, :].reshape(3, 3)

        # Enforcing the internal constraint that two singular values must be
        # non-zero and one must be zero.
        U, S, V = np.linalg.svd(F)
        S[0] = S[1] = (S[0] + S[1]) / 2.0
        S[2] = 0
        self.params = U @ np.diag(S) @ V
        return True

    def residuals(self, src, dst):
        # Compute the Sampson distance.
        src_homogeneous = np.column_stack([src, np.ones(src.shape[0])])
        dst_homogeneous = np.column_stack([dst, np.ones(dst.shape[0])])

        F_src = self.params @ src_homogeneous.T
        Ft_dst = self.params.T @ dst_homogeneous.T

        dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)

        return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2
                                           + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)


def extractFeatures(img):
    orb = cv2.ORB_create()
    # detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                                  3000, qualityLevel=0.01, minDistance=7)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    kps, des = orb.compute(img, kps)

    # return pts and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


def match_frames(kps1, des1, kps2, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    
    # # Lowe's ratio test
    # ret = []
    # idx1, idx2 = [], []
    # idx1s, idx2s = set(), set()

    # for m, n in matches:
    #     if m.distance < 0.75*n.distance:
    #         p1 = kps1[m.queryIdx]
    #         p2 = kps2[m.trainIdx]

    #     # be within orb distance 32
    #     if m.distance < 32:
    #         # keep around indices
    #         # TODO: refactor this to not be O(N^2)
    #         if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
    #             idx1.append(m.queryIdx)
    #             idx2.append(m.trainIdx)
    #             idx1s.add(m.queryIdx)
    #             idx2s.add(m.trainIdx)
    #             ret.append((p1, p2))

    # # no duplicates
    # assert(len(set(idx1)) == len(idx1))
    # assert(len(set(idx2)) == len(idx2))

    # assert len(ret) >= 8
    # ret = np.array(ret)
    # idx1 = np.array(idx1)
    # idx2 = np.array(idx2)


# ********************************************************************
    matches = match_ratio_test(matches)
    
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kps1[match[0].queryIdx]
        points2[i, :] = kps2[match[0].trainIdx]
    indices1, indices2 = get_match_indices(matches)
    print(points1.shape)
    print(points2.shape)
    
# ********************************************************************

    # fit matrix
    model, inliers = ransac((points1, points2),
                            EssentialMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.2,
                            max_trials=100)
    
    print("Matches:  %d -> %d -> %d -> %d" % (len(des1), len(matches),
                                              len(inliers), sum(inliers)))
    # return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)
    return indices1[inliers], indices2[inliers], fundamentalToRt(model.params)
