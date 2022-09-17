from paz import processors as pr
from backend import detect_SIFT_features
from backend import brute_force_matcher
from backend import find_homography_RANSAC


class DetecetSiftFeatures(pr.Processor):
    def __init__(self):
        super(DetecetSiftFeatures, self).__init__()

    def call(self, image):
        return detect_SIFT_features(image)


class BruteForceMatcher(pr.Processor):
    def __init__(self, k=2):
        self.k = k
        super(BruteForceMatcher, self).__init__()

    def call(self, descriptor1, descriptor2):
        return brute_force_matcher(descriptor1, descriptor2, self.k)


class FindHomographyRANSAC(pr.Processor):
    def __init__(self):
        super(FindHomographyRANSAC, self).__init__()

    def call(self, points1, points2):
        return find_homography_RANSAC(points1, points2)
