from paz import processors as pr
from processors import DetecetSiftFeatures, FindHomographyRANSAC, RecoverPose
from processors import FindFundamentalMatrix, TriangulatePoints
from processors import BruteForceMatcher
import cv2
import numpy as np


class FeatureDetector(pr.Processor):
    def __init__(self, detector=DetecetSiftFeatures(),
                 matcher=BruteForceMatcher(), draw=True):
        super(FeatureDetector, self).__init__()
        self.draw = draw
        self.detector = detector
        self.matcher = matcher
        self.warp = pr.WrapOutput(['keypoint_image', 'match_image',
                                   'match_points', 'keypoints', 'discriptors'])

    def call(self, images):
        image1, image2 = images
        keypoints1, descriptor1 = self.detector(image1)
        keypoints2, descriptor2 = self.detector(image2)

        if self.draw:
            result1 = cv2.drawKeypoints(image1, keypoints1, outImage=None)
            result2 = cv2.drawKeypoints(image2, keypoints2, outImage=None)
            keypoint_images = np.concatenate((result1, result2), axis=1)

        matches = self.matcher(descriptor1, descriptor2)

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match[0].queryIdx].pt
            points2[i, :] = keypoints2[match[0].trainIdx].pt

        if self.draw:
            match_image = cv2.drawMatchesKnn(
                image1, keypoints1, image2, keypoints2, matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return self.warp(keypoint_images, match_image, [points1, points2],
                         [keypoints1, keypoints2], [descriptor1, descriptor2])


class ComputeHomography(pr.Processor):
    def __init__(self, draw=True):
        super(ComputeHomography, self).__init__()
        self.draw = draw
        self.feature_detector = FeatureDetector()
        self.compute_homography = FindHomographyRANSAC()
        self.warp = pr.WrapOutput(['image', 'homography'])

    def call(self, images):
        image1, image2 = images
        H_, W_ = image2.shape[:2]
        matches = self.feature_detector(images)
        points1, points2 = matches['match_points']
        H, mask = self.compute_homography(points1, points2)
        image_ = cv2.warpPerspective(image2, H, (W_, H_))
        image = np.concatenate((image1, image_), axis=1)
        print(H)
        return self.warp(image, H)


class StructureFromMotion(pr.Processor):
    def __init__(self, camera_intrinsics, draw=True):
        super(StructureFromMotion, self).__init__()
        self.K = camera_intrinsics
        self.draw = draw
        self.feature_detector = FeatureDetector()
        self.compute_fundamental_matrix = FindFundamentalMatrix()
        self.recover_pose = RecoverPose(self.K)
        self.triangulate_points = TriangulatePoints()
        self.warp = pr.WrapOutput(['points3D'])
        self.projection_matrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0]])

    def call(self, images):
        matches = self.feature_detector(images)
        points1, points2 = matches['match_points']
        F, mask = self.compute_fundamental_matrix(points1, points2)
        E = self.K.T @ (F @ self.K)
        points, R, t, mask = self.recover_pose(E, points1, points2)
        # print(points, R, t)
        # print(E)
        P1 = self.K @ self.projection_matrix
        P2 = R @ self.projection_matrix
        P2[:3, 3] = t.ravel()
        P2 = self.K @ P2
        print(P2)

        points3D = self.triangulate_points(P1, P2, points1.T, points2.T)
        print(points3D)

        return self.warp(points3D)
