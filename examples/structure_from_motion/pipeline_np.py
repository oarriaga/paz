import numpy as np
import matplotlib.pyplot as plt
from paz import processors as pr
from processors import DetecetSIFTFeatures, RecoverPose
from processors import ComputeEssentialMatrix
from processors import FindFundamentalMatrix, TriangulatePoints
from processors import FLANNMatcher, SolvePnP
from processors import ComputeFundamentalMatrixRANSAC as RANSAC
from backend import match_ratio_test, get_match_points, get_match_indices
from backend import contruct_projection_matrix
from backend import triangulate_points_np


class MatchFeatures(pr.Processor):
    def __init__(self, matcher=FLANNMatcher(k=2), ratio_test=True):
        super(MatchFeatures, self).__init__()
        self.matcher = matcher
        self.ratio_test = ratio_test

    def call(self,  des1, des2, ratio=0.75):
        matches = self.matcher(des1, des2)
        if self.ratio_test:
            matches = match_ratio_test(matches, ratio)
        return matches


def plot_3D_keypoints(keypoints3D):
    ax = plt.axes(projection='3d')
    ax.view_init(-160, -80)
    ax.figure.canvas.set_window_title('Minimal hand 3D plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for arg in range(len(keypoints3D)):
        points = keypoints3D[arg]
        xs, ys, zs = np.split(points, 3, axis=1)
        ax.scatter(xs, ys, zs, s=5)
    plt.show()


class FindCorrespondances(pr.Processor):
    def __init__(self):
        super(FindCorrespondances, self).__init__()
        self.detector = DetecetSIFTFeatures()
        self.match_features = MatchFeatures()
        self.ransac_filter = RANSAC(residual_thresh=2, max_trials=100)

    def call(self, base_features, image):
        base_kps, base_des = base_features
        keypoints1, descriptor1 = self.detector(image)
        matches = self.match_features(base_des, descriptor1)
        points1, points2 = get_match_points(base_kps, keypoints1, matches)
        indices = get_match_indices(matches)
        model, inliers = self.ransac_filter(points1, points2)
        p1_inliers = points1[inliers]
        p2_inliers = points2[inliers]
        base_indices_inliers = indices[0][inliers]
        return p1_inliers, p2_inliers, base_indices_inliers


class ProjectionFromCorrespondances(pr.Processor):
    def __init__(self, camera_intrinsics):
        super(ProjectionFromCorrespondances, self).__init__()
        self.K = camera_intrinsics
        self.solve_pnp = SolvePnP(camera_intrinsics)
        self.rotation_vector_to_matrix = pr.RotationVectorToRotationMatrix()
        self.projection_matrix = np.eye(3, 4)

    def call(self, points3D, points2D):
        _, rotation, translation = self.solve_pnp(points3D, points2D)
        rotation = self.rotation_vector_to_matrix(rotation)
        P3 = rotation @ self.projection_matrix
        P3[:3, 3] = translation.ravel()
        P3 = self.K @ P3


class InitializeSFM(pr.Processor):
    def __init__(self, camera_intrinsics, draw=True):
        super(InitializeSFM, self).__init__()
        self.K = camera_intrinsics
        self.draw = draw
        self.detector = DetecetSIFTFeatures()
        self.match_features = MatchFeatures()
        self.ransac_filter = RANSAC()
        self.compute_fundamental_matrix = FindFundamentalMatrix()
        self.compute_essential_matrix = ComputeEssentialMatrix(self.K)
        self.recover_pose = RecoverPose(self.K)
        self.warp = pr.WrapOutput(['points3D', 'base_features', 'P2'])

    def call(self, images):
        image1, image2 = images[:2]
        kps1, des1 = self.detector(image1)
        kps2, des2 = self.detector(image2)
        matches = self.match_features(des1, des2)
        points1, points2 = get_match_points(kps1, kps2, matches)
        indices1, indices2 = get_match_indices(matches)
        model, inliers = self.ransac_filter(points1, points2)

        p1_inliers = points1[inliers]
        p2_inliers = points2[inliers]
        print('Number of inliers', p1_inliers.shape[0])

        # F = model.params
        F, _ = self.ransac_filter(p1_inliers, p2_inliers)
        E = self.compute_essential_matrix(F)
        rotation, translation = self.recover_pose(E, p1_inliers, p2_inliers)

        P1 = self.K @ np.eye(3, 4)
        P2 = contruct_projection_matrix(rotation, translation)
        P2 = self.K @ P2

        points3D = triangulate_points_np(P1, P2, p1_inliers, p2_inliers)

        base_kps = kps2[indices2[inliers]]
        base_des = des2[indices2[inliers]]
        return self.warp(np.array(points3D), [base_kps, base_des], P2)


class StructureFromMotion(pr.Processor):
    def __init__(self, camera_intrinsics):
        super(StructureFromMotion, self).__init__()
        self.K = camera_intrinsics
        self.initialize_sfm = InitializeSFM(camera_intrinsics)
        self.detector = DetecetSIFTFeatures()
        self.match_features = MatchFeatures()
        self.ransac_filter = RANSAC()
        self.find_correspondences = FindCorrespondances()
        self.solve_pnp = SolvePnP(camera_intrinsics)
        self.rotation_vector_to_matrix = pr.RotationVectorToRotationMatrix()
        self.triangulate_points = TriangulatePoints()
        self.points3D = []
        self.projection_matrix = np.eye(3, 4)

    def call(self, images):
        inference = self.initialize_sfm(images[:2])
        points3D = inference['points3D']
        self.points3D.append(points3D)
        base_features = inference['base_features']
        P2 = inference['P2']

        for arg in range(len(images)-2):
            keypoints1, descriptor1 = self.detector(images[arg + 1])
            keypoints2, descriptor2 = self.detector(images[arg + 2])

            p1_inliers, p2_inliers, indices = self.find_correspondences(
                base_features, images[arg + 2])

            _, rotation, translation = self.solve_pnp(points3D[indices],
                                                      p1_inliers)

            rotation = self.rotation_vector_to_matrix(rotation)
            P3 = contruct_projection_matrix(rotation, translation)
            P3 = self.K @ P3

            matches = self.match_features(descriptor1, descriptor2)
            indices = get_match_indices(matches)
            points1, points2 = get_match_points(keypoints1, keypoints2,
                                                matches)
            model, inliers = self.ransac_filter(points1, points2)

            p1_inliers = points1[inliers]
            p2_inliers = points2[inliers]
            print('Number of inliers', p1_inliers.shape[0])

            points3D = triangulate_points_np(P2, P3, p1_inliers, p2_inliers)

            self.points3D.append(points3D)
            P2 = P3.copy()

            arg = indices[1][inliers]
            base_features = [keypoints2[arg], descriptor2[arg]]
        plot_3D_keypoints(self.points3D)
        return self.points3D
