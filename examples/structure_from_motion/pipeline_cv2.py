import cv2
import numpy as np
import matplotlib.pyplot as plt
from paz import processors as pr
from processors import RecoverPose
from processors import FLANNMatcher
from processors import DetecetSIFTFeatures
from processors import ComputeEssentialMatrix
from processors import FindFundamentalMatrix, TriangulatePoints
from processors import SolvePnP
from processors import ComputeFundamentalMatrixRANSAC as RANSAC
from backend import match_ratio_test, get_match_points, get_match_indices
from backend import contruct_projection_matrix
from backend import extract_keypoints_RGB
from bundle_adjustment import local_bundle_adjustment
from paz.backend.groups import to_affine_matrix

ransac_thresh = 0.1
ransac_corr_thresh = 0.5
match_ratio = 0.3


class MatchFeatures(pr.Processor):
    def __init__(self, matcher=FLANNMatcher(k=2), ratio_test=True):
        super(MatchFeatures, self).__init__()
        self.matcher = matcher
        self.ratio_test = ratio_test

    def call(self,  des1, des2, ratio=match_ratio):
        matches = self.matcher(des1, des2)
        if self.ratio_test:
            matches = match_ratio_test(matches, ratio)
        return matches


def plot_3D_keypoints(keypoints3D, colors, camera_position):
    ax = plt.axes(projection='3d')
    ax.view_init(-160, -80)
    ax.figure.canvas.set_window_title('Minimal hand 3D plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for arg in range(len(keypoints3D)):
        points3D = keypoints3D[arg]
        print('points3D', points3D)
        print('points3D.shape', points3D.shape)
        color = np.array(colors[arg])
        xs, ys, zs = np.split(points3D, 3, axis=1)
        ax.scatter(xs, ys, zs, s=5, c=color/255)
    for arg in range(len(camera_position)):
        ax.scatter(camera_position[arg][0], camera_position[arg][1],
                   camera_position[arg][2], s=50, marker='^', c='r')
    plt.show()


class FindCorrespondances(pr.Processor):
    def __init__(self):
        super(FindCorrespondances, self).__init__()
        self.detector = DetecetSIFTFeatures()
        self.match_features = MatchFeatures()
        self.ransac_filter = RANSAC(residual_thresh=ransac_corr_thresh,
                                    max_trials=100)

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
        self.ransac_filter = RANSAC(residual_thresh=ransac_thresh)
        self.compute_fundamental_matrix = FindFundamentalMatrix(
            ransacReprojThreshold=ransac_thresh)
        self.compute_essential_matrix = ComputeEssentialMatrix(self.K)
        self.recover_pose = RecoverPose(self.K)
        self.triangulate_points = TriangulatePoints()
        self.initial_transform = np.eye(4)
        self.warp = pr.WrapOutput(['points3D', 'base_features',
                                   'P2', 'colors', 'camera_position'])

    def call(self, images):
        image1, image2 = images[:2]
        keypoints1, descriptors1 = self.detector(image1)
        keypoints2, descriptors2 = self.detector(image2)
        matches = self.match_features(descriptors1, descriptors2)
        points1, points2 = get_match_points(keypoints1, keypoints2, matches)
        indices1, indices2 = get_match_indices(matches)
        model, inliers = self.ransac_filter(points1, points2)
        p1_inliers = points1[inliers]
        p2_inliers = points2[inliers]

        colors = extract_keypoints_RGB(image1, p1_inliers)
        print('Number of inliers', p1_inliers.shape[0])

        F, _ = self.compute_fundamental_matrix(p1_inliers, p2_inliers)
        E = self.compute_essential_matrix(F)
        rotation, translation = self.recover_pose(E, p1_inliers, p2_inliers)
        camera_position = np.matmul(rotation, translation)

        transform = to_affine_matrix(rotation, translation)

        P1 = self.K @ np.eye(3, 4)
        P2 = contruct_projection_matrix(rotation, translation)
        P2 = self.K @ P2
        points3D = self.triangulate_points(P1, P2, p1_inliers.T, p2_inliers.T)
        optimized_points3D, _ = local_bundle_adjustment(rotation, translation,
                                                        points3D, p2_inliers,
                                                        self.K)

        current_transform = np.matmul(self.initial_transform, transform)
        optimized_points3D = np.matmul(current_transform[:3, :3],
                                       optimized_points3D.T).T + \
            current_transform[:3, 3]

        self.initial_transform = current_transform

        base_kps = keypoints2[indices2[inliers]]
        base_des = descriptors2[indices2[inliers]]
        return self.warp(np.array(optimized_points3D), [base_kps, base_des],
                         P2, colors, camera_position)


class StructureFromMotion(pr.Processor):
    def __init__(self, camera_intrinsics):
        super(StructureFromMotion, self).__init__()
        self.camera_intrinsics = camera_intrinsics
        self.sfm = InitializeSFM(self.camera_intrinsics)

    def call(self, images):
        points3D = []
        colors = []
        camera_positions = []
        for arg in range(len(images)-1):
            inference = self.sfm([images[arg], images[arg + 1]])
            points3D.append(inference['points3D'])
            colors.append(inference['colors'])
            camera_positions.append(inference['camera_position'])
        plot_3D_keypoints(points3D, colors, camera_positions)
        return points3D


class FeatureDetector(pr.Processor):
    def __init__(self, detector=DetecetSIFTFeatures(),
                 matcher=MatchFeatures(), draw=True):
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


# class StructureFromMotion(pr.Processor):
#     def __init__(self, camera_intrinsics):
#         super(StructureFromMotion, self).__init__()
#         self.K = camera_intrinsics
#         self.initialize_sfm = InitializeSFM(camera_intrinsics)
#         self.detector = DetecetSIFTFeatures()
#         self.match_features = MatchFeatures()
#         self.ransac_filter = RANSAC(residual_thresh=ransac_thresh)
#         self.find_correspondences = FindCorrespondances()
#         self.solve_pnp = SolvePnP(camera_intrinsics)
#         self.rotation_vector_to_matrix = pr.RotationVectorToRotationMatrix()
#         self.triangulate_points = TriangulatePoints()
#         self.points3D = []
#         self.colors = []
#         self.camera_position = []
#         self.projection_matrix = np.eye(3, 4)

#     def call(self, images):
#         inference = self.initialize_sfm(images[:2])
#         points3D = inference['points3D']
#         self.points3D.append(points3D)
#         self.colors.append(inference['colors'])
#         base_features = inference['base_features']
#         P2 = inference['P2']

#         for arg in range(len(images)-2):
#             # print('Processing image', arg)
#             image1 = images[arg + 1]
#             image2 = images[arg + 2]
#             keypoints1, descriptor1 = self.detector(image1)
#             keypoints2, descriptor2 = self.detector(image2)
#             p1_inliers, p2_inliers, indices = self.find_correspondences(
#                 base_features, image2)
#             _, rotation, translation = self.solve_pnp(points3D[indices],
#                                                       p1_inliers)
#             rotation = self.rotation_vector_to_matrix(rotation)
#             self.camera_position.append(np.matmul(rotation, translation))
#             # print('Rotation', rotation)
#             P3 = contruct_projection_matrix(rotation, translation)
#             P3 = self.K @ P3

#             matches = self.match_features(descriptor1, descriptor2)
#             indices = get_match_indices(matches)
#             points1, points2 = get_match_points(keypoints1, keypoints2,
#                                                 matches)
#             model, inliers = self.ransac_filter(points1, points2)
#             p1_inliers = points1[inliers]
#             p2_inliers = points2[inliers]
#             colors = extract_keypoints_RGB(image1, p1_inliers)

#             print('Number of inliers for image:', arg, p1_inliers.shape[0])
#             points3D = self.triangulate_points(P2, P3, p1_inliers.T,
#                                                p2_inliers.T)
#             # points3D, inliers = remove_outliers_cKDTree(points3D)
#             optimized_points3D, _ = local_bundle_adjustment(
#                 rotation, translation, points3D, p2_inliers, self.K)
#             self.points3D.append(optimized_points3D)
#             self.colors.append(colors)
#             P2 = P3.copy()

#             arg = indices[1][inliers]
#             base_features = [keypoints2[arg], descriptor2[arg]]
#         plot_3D_keypoints(self.points3D, self.colors, self.camera_position)
#         return self.points3D
