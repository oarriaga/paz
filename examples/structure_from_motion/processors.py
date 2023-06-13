from paz import processors as pr
from backend import recover_pose_cv
from backend import triangulate_points_cv
from backend import detect_SIFT_features
from backend import FLANN_matcher
from backend import brute_force_matcher
from backend import find_homography_RANSAC_cv
from backend import find_fundamental_matrix_cv
from backend import compute_essential_matrix
from backend import estimate_homography_ransac_np
from backend import estimate_fundamental_matrix_ransac_np
from backend import triangulate_points_np
from backend import compute_fundamental_matrix_np
from paz.backend.keypoints import solve_PnP_RANSAC
from skimage_ransac import ransac
from transform import FundamentalMatrixTransform


class DetecetSIFTFeatures(pr.Processor):
    """
    Detects SIFT features

    Returns:
        keypoints: numpy array of shape (num_keypoints, 2)
                   containing the keypoints of the image
        descriptors: numpy array of shape (num_keypoints, 128)
                     containing the descriptors of the image
    """
    def __init__(self):
        super(DetecetSIFTFeatures, self).__init__()

    def call(self, image):
        return detect_SIFT_features(image)


class BruteForceMatcher(pr.Processor):
    """
    Brute force matcher for SIFT features

    Arguments:
        k: int
           number of nearest neighbors to return

    Returns:
        matches: numpy array of shape (num_matches, 2)
                 containing the matches between the descriptor of two images
    """
    def __init__(self, k=2):
        self.k = k
        super(BruteForceMatcher, self).__init__()

    def call(self, descriptor1, descriptor2):
        return brute_force_matcher(descriptor1, descriptor2, self.k)


class FLANNMatcher(pr.Processor):
    """
    FLANN matcher for SIFT features

    Arguments:
        k: int
           number of nearest neighbors to return

    Returns:
        matches: numpy array of shape (num_matches, 2)
                 containing the matches between the descriptor of two images
    """
    def __init__(self, k=2):
        self.k = k
        super(FLANNMatcher, self).__init__()

    def call(self, descriptor1, descriptor2):
        return FLANN_matcher(descriptor1, descriptor2, self.k)


class FindHomographyRANSAC(pr.Processor):
    """
    Finds homography between two images using RANSAC

    Arguments:
        ransacReprojThreshold : float
                                threshold for the reprojection error
        max_iterations: int
                        maximum number of iterations to run RANSAC
    Returns:
        homography: numpy array of shape (3, 3)
                    containing the homography between the two images"""
    def __init__(self, ransacReprojThreshold=0.5, maxIters=1000):
        super(FindHomographyRANSAC, self).__init__()
        self.ransacReprojThreshold = ransacReprojThreshold
        self.maxIters = maxIters

    def call(self, points1, points2):
        return find_homography_RANSAC_cv(
            points1, points2, self.ransacReprojThreshold, self.maxIters)


class FindFundamentalMatrix(pr.Processor):
    """
    Computes fundamental matrix between two points

    Arguments:
        ransacReprojThreshold: float
                               threshold for the reprojection error
        confidence: float
                    confidence for the RANSAC algorithm
        maxIters: int
                  maximum number of iterations to run RANSAC

    Returns:
        fundamental_matrix: numpy array of shape (3, 3)
                           containing the fundamental matrix between two images
    """
    def __init__(self, ransacReprojThreshold=0.5, confidence=0.99,
                 maxIters=1000):
        super(FindFundamentalMatrix, self).__init__()
        self.ransacReprojThreshold = ransacReprojThreshold
        self.confidence = confidence
        self.maxIters = maxIters

    def call(self, points1, points2):
        return find_fundamental_matrix_cv(
            points1, points2, self.ransacReprojThreshold, self.confidence,
            self.maxIters)


class ComputeEssentialMatrix(pr.Processor):
    """
    Computes essential matrix from fundamental matrix

    Arguments:
        camera_intrinsics: numpy array of shape (3, 3)
                           containing the camera intrinsics

    Returns:
        essential_matrix: numpy array of shape (3, 3)
                          containing the essential matrix between two images
    """
    def __init__(self, camera_intrinsics):
        super(ComputeEssentialMatrix, self).__init__()
        self.camera_intrinsics = camera_intrinsics

    def call(self, fundamental_matrix):
        return compute_essential_matrix(fundamental_matrix,
                                        self.camera_intrinsics)


class RecoverPose(pr.Processor):
    """
    Recovers the pose of a second camera from the essential matrix

    Arguments:
        camera_intrinsics: numpy array of shape (3, 3)
                           containing the camera intrinsics

    Returns:
        rotation: numpy array of shape (3, 3)
                  containing the rotation matrix
        translation: numpy array of shape (3, 1)
                    containing the translation vector
    """
    def __init__(self, camera_intrinsics):
        super(RecoverPose, self).__init__()
        self.camera_intrinsics = camera_intrinsics

    def call(self, essensial_matrix, points1, points2):
        return recover_pose_cv(essensial_matrix, points1,
                               points2, self.camera_intrinsics)


class TriangulatePoints(pr.Processor):
    """
    Triangulate a set of points from two cameras.

    Arguments:
        P1: numpy array of shape (3, 4)
            containing the projection matrix of the first camera
        P2: numpy array of shape (3, 4)
            containing the projection matrix of the second camera
        points1: numpy array of shape (num_points, 2)
                 containing the points in the first image
        points2: numpy array of shape (num_points, 2)
                 containing the points in the second image

    Returns:
        points3D: numpy array of shape (num_points, 3)
                  containing the triangulated points"""
    def __init__(self):
        super(TriangulatePoints, self).__init__()

    def call(self, P1, P2, points1, points2):
        return triangulate_points_cv(P1, P2, points1, points2)


class SolvePnP(pr.Processor):
    """
    Solves the Perspective-n-Point problem using RANSAC

    Arguments:
        camera_intrinsics: numpy array of shape (3, 3)
                           containing the camera intrinsics

    Returns:
        rotation: numpy array of shape (3,)
                  containing the rotation matrix in axis-angle form
        translation: numpy array of shape (3,)
                     containing the translation vector
    """
    def __init__(self, camera_intrinsics):
        super(SolvePnP, self).__init__()
        self.camera_intrinsics = camera_intrinsics

    def call(self, points3D, points2D):
        return solve_PnP_RANSAC(points3D, points2D, self.camera_intrinsics,
                                inlier_threshold=5)


class EstimateHomographyRANSAC(pr.Processor):
    """
    Estimates homography between two images using RANSAC

    Arguments:
        min_samples: int
                     minimum number of samples to fit the model
        residual_thresh: float
                         threshold for the reprojection error
        max_trials: int
                    maximum number of iterations to run RANSAC

    Returns:
        homography: numpy array of shape (3, 3)
                    containing the homography between the two set of points
        inlieres: numpy array of shape (num_inliers,)
                  containing the inlier indices
    """
    def __init__(self, min_samples=8, residual_thresh=0.5, max_trials=1000):
        super(EstimateHomographyRANSAC, self).__init__()
        self.min_samples = min_samples
        self.residual_thresh = residual_thresh
        self.max_trials = max_trials

    def call(self, points1, points2):
        homography, inlieres = estimate_homography_ransac_np(
            points1, points2, self.min_samples, self.residual_thresh,
            self.max_trials)
        return homography, inlieres


class SkimageRANSAC(pr.Processor):
    """
    Estimates homography between two images using skimage RANSAC

    Arguments:
        min_samples: int
                     minimum number of samples to fit the model
        residual_thresh: float
                         threshold for the reprojection error
        max_trials: int
                    maximum number of iterations to run RANSAC

    Returns:
        homography: numpy array of shape (3, 3)
                    containing the homography between the two set of points
        inlieres: numpy array of shape (num_inliers,)
                  containing the inlier indices
    """
    def __init__(self, min_samples=8, residual_thresh=0.5, max_trials=1000):
        super(SkimageRANSAC, self).__init__()
        self.model_class = FundamentalMatrixTransform
        self.min_samples = min_samples
        self.residual_thresh = residual_thresh
        self.max_trials = max_trials

    def call(self, points1, points2):
        model, inliers = ransac(
            (points1, points2), self.model_class, self.min_samples,
            self.residual_thresh, max_trials=self.max_trials)
        return model, inliers


class ComputeFundamentalMatrixRANSAC(pr.Processor):
    """
    Estimates fundamental matrix between two images using RANSAC

    Arguments:
        min_samples: int
                     minimum number of samples to fit the model
        residual_thresh: float
            threshold for the reprojection error
        max_trials: int
                    maximum number of iterations to run RANSAC

    Returns:
        fundamental_matrix: numpy array of shape (3, 3)
                            containing the fundamental matrix between the
                            two set of points
        inlieres: numpy array of shape (num_inliers,)
                  containing the inlier indices
    """
    def __init__(self, min_samples=8, residual_thresh=0.5, max_trials=1000):
        super(ComputeFundamentalMatrixRANSAC, self).__init__()
        self.min_samples = min_samples
        self.residual_thresh = residual_thresh
        self.max_trials = max_trials

    def call(self, points1, points2):
        return estimate_fundamental_matrix_ransac_np(points1, points2)
