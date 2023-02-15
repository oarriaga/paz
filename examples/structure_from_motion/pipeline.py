from paz import processors as pr
from processors import DetecetSiftFeatures, RecoverPose
from processors import FindHomographyRANSAC, ComputeEssentialMatrix
from processors import ComputeFundamentalMatrix, TriangulatePoints
from processors import BruteForceMatcher, SolvePnP
from backend import match_ratio_test, get_match_points, get_match_indices
# from backend import get_match_descriptors
import cv2
import numpy as np
import matplotlib.pyplot as plt


class MatchFeatures(pr.Processor):
    def __init__(self, matcher=BruteForceMatcher(k=2), ratio_test=True):
        super(MatchFeatures, self).__init__()
        self.matcher = matcher
        self.ratio_test = ratio_test

    def call(self,  descriptor1, descriptor2):
        matches = self.matcher(descriptor1, descriptor2)
        if self.ratio_test:
            matches = match_ratio_test(matches)
        # points = get_match_points(keypoints1, keypoints2, matches)
        # indices = get_match_indices(matches)
        # return points, indices
        return matches


class InitializeSFM(pr.Processor):
    def __init__(self, camera_intrinsics, draw=True):
        super(InitializeSFM, self).__init__()
        self.K = camera_intrinsics
        self.draw = draw
        self.detector = DetecetSiftFeatures()
        self.match_features = MatchFeatures()
        self.compute_fundamental_matrix = ComputeFundamentalMatrix()
        self.compute_essential_matrix = ComputeEssentialMatrix(self.K)
        self.recover_pose = RecoverPose(self.K)
        self.triangulate_points = TriangulatePoints()
        self.warp = pr.WrapOutput(['points3D', 'base_features', 'P2'])
        self.projection_matrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0]])

    def call(self, images):
        # take two images
        image1, image2 = images

        # extract features
        keypoints1, descriptor1 = self.detector(image1)
        keypoints2, descriptor2 = self.detector(image2)

        # match features
        matches = self.match_features(descriptor1, descriptor2)
        points = get_match_points(keypoints1, keypoints2, matches)
        indices = get_match_indices(matches)
        points1, points2 = points

        # compute fundamental matrix
        fundamental_matrix, mask = self.compute_fundamental_matrix(points1,
                                                                   points2)

        # compute essential matrix
        essential_matrix = self.compute_essential_matrix(fundamental_matrix)

        # recover camera pose
        points, rotation, translation, mask = self.recover_pose(
            essential_matrix, points1, points2)

        # get projection matrix
        P1 = self.K @ self.projection_matrix
        P2 = rotation @ self.projection_matrix
        P2[:3, 3] = translation.ravel()
        P2 = self.K @ P2

        # triangulation
        points3D = self.triangulate_points(P1, P2, points1.T, points2.T)
        # print(points3D)
        keypoints2 = np.array(keypoints2)
        descriptor2 = np.array(descriptor2)
        return self.warp(points3D, [keypoints2[indices[1]],
                                    descriptor2[indices[1]]], P2)


def plot_3D_keypoints(keypoints3D):
    ax = plt.axes(projection='3d')
    ax.view_init(-160, -80)
    ax.figure.canvas.set_window_title('Minimal hand 3D plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for arg in range(len(keypoints3D)):
        points = keypoints3D[arg]
        print(points.shape)
        xs, ys, zs = np.split(points, 3, axis=1)
        ax.scatter(xs, ys, zs, s=5)
        # break
    plt.show()


class FindCorrespondances(pr.Processor):
    def __init__(self):
        super(FindCorrespondances, self).__init__()
        self.detector = DetecetSiftFeatures()
        self.match_features = MatchFeatures()

    def call(self, base_features, image):
        keypoints1, descriptor1 = self.detector(image)
        # points, indices = self.match_features(
        #     base_features[0], base_features[1], keypoints1, descriptor1)

        matches = self.match_features(
            base_features[0], base_features[1], keypoints1, descriptor1)
        points = get_match_points(base_features[0], keypoints1, matches)
        indices = get_match_indices(matches)
        return points, indices


class ProjectionFromCorrespondances(pr.Processor):
    def __init__(self, camera_intrinsics):
        super(ProjectionFromCorrespondances, self).__init__()
        self.K = camera_intrinsics
        self.solve_pnp = SolvePnP(camera_intrinsics)
        self.rotation_vector_to_matrix = pr.RotationVectorToRotationMatrix()
        self.projection_matrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0]])

    def call(self, points3D, points2D):
        _, rotation, translation = self.solve_pnp(points3D, points2D)
        rotation = self.rotation_vector_to_matrix(rotation)
        P3 = rotation @ self.projection_matrix
        P3[:3, 3] = translation.ravel()
        P3 = self.K @ P3


class StructureFromMotion(pr.Processor):
    def __init__(self, camera_intrinsics):
        super(StructureFromMotion, self).__init__()
        self.K = camera_intrinsics
        self.initialize_sfm = InitializeSFM(camera_intrinsics)
        self.detector = DetecetSiftFeatures()
        self.match_features = MatchFeatures()
        self.find_correspondences = FindCorrespondances()
        self.solve_pnp = SolvePnP(camera_intrinsics)
        self.rotation_vector_to_matrix = pr.RotationVectorToRotationMatrix()
        self.triangulate_points = TriangulatePoints()
        self.points3D = []
        self.projection_matrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0]])

    def call(self, images):
        # points3D: 3D keypoints from two images
        # base feature: mathing keypoints and descriptors from image 2
        # P2: pojection matrix
        inference = self.initialize_sfm(images[:2])
        points3d = np.array(inference['points3D'])
        self.points3D.append(points3d)
        base_features = inference['base_features']
        P2 = inference['P2']
        # points, indices = self.find_correspondences(base_features, images[3])

        for arg in range(len(images)-2):

            keypoints1, descriptor1 = self.detector(images[arg + 1])
            keypoints2, descriptor2 = self.detector(images[arg + 2])

            matches = self.match_features(base_features[1], descriptor2)
            points = get_match_points(base_features[0], keypoints2, matches)
            indices = get_match_indices(matches)

            _, rotation, translation = self.solve_pnp(points3d[indices[0]],
                                                      points[1])
            rotation = self.rotation_vector_to_matrix(rotation)
            P3 = rotation @ self.projection_matrix
            P3[:3, 3] = translation.ravel()
            P3 = self.K @ P3

            matches = self.match_features(descriptor1, descriptor2)
            points = get_match_points(keypoints1, keypoints2, matches)
            indices = get_match_indices(matches)
            print(indices)
            print('********************************')
            points1, points2 = points

            points3d = self.triangulate_points(P2, P3, points1.T, points2.T)

            self.points3D.append(points3d)

            keypoints2 = np.array(keypoints2)
            descriptor2 = np.array(descriptor2)
            base_features = [keypoints2[indices[1]], descriptor2[indices[1]]]
            print(P3)
            print('##################')
            P2 = P3.copy()
            # break
        # plot_3D_keypoints(self.points3D)
        return self.points3D


# todo
# RANSAC implemenatation in numpy 
# eight point algorithm implementation in numpy.
