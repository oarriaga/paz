from paz import processors as pr
from processors import DetecetSiftFeatures, FindHomographyRANSAC
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
        H, W = image2.shape[:2]
        matches = self.feature_detector(images)
        points1, points2 = matches['match_points']
        homography, mask = self.compute_homography(points1, points2)
        image_ = cv2.warpPerspective(image2, homography, (W, H))
        image = np.concatenate((image1, image_), axis=1)
        return self.warp(image, homography)
