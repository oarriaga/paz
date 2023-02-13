from paz import processors as pr
from backend1 import detect_SIFT_features, recover_pose, triangulate_points
from backend1 import brute_force_matcher
from backend1 import find_homography_RANSAC
from backend1 import find_fundamental_matrix, compute_essential_matrix
from paz.backend.keypoints import solve_PnP_RANSAC


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


class ComputeFundamentalMatrix(pr.Processor):
    def __init__(self):
        super(ComputeFundamentalMatrix, self).__init__()

    def call(self, points1, points2):
        return find_fundamental_matrix(points1, points2)


class ComputeEssentialMatrix(pr.Processor):
    def __init__(self, camera_intrinsics):
        super(ComputeEssentialMatrix, self).__init__()
        self.camera_intrinsics = camera_intrinsics

    def call(self, fundamental_matrix):
        return compute_essential_matrix(fundamental_matrix,
                                        self.camera_intrinsics)


class RecoverPose(pr.Processor):
    def __init__(self, camera_intrinsics):
        super(RecoverPose, self).__init__()
        self.camera_intrinsics = camera_intrinsics

    def call(self, essensial_matrix, points1, points2):
        return recover_pose(essensial_matrix, points1,
                            points2, self.camera_intrinsics)


class TriangulatePoints(pr.Processor):
    def __init__(self):
        super(TriangulatePoints, self).__init__()

    def call(self, P1, P2, points1, points2):
        return triangulate_points(P1, P2, points1, points2)


class SolvePnP(pr.Processor):
    def __init__(self, camera_intrinsics):
        super(SolvePnP, self).__init__()
        self.camera_intrinsics = camera_intrinsics

    def call(self, points3D, points2D):
        return solve_PnP_RANSAC(points3D, points2D, self.camera_intrinsics,
                                inlier_threshold=5)
