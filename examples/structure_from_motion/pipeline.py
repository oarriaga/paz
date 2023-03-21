import numpy as np
import matplotlib.pyplot as plt
from paz import processors as pr
from processors import DetecetSiftFeatures, RecoverPose
from processors import ComputeEssentialMatrix
from processors import ComputeFundamentalMatrix, TriangulatePoints
from processors import BruteForceMatcher, SolvePnP
from backend import match_ratio_test, get_match_points, get_match_indices
from backend import detect_ORB_fratures, brute_force_matcher
from backend import detect_SIFT_features
from skimage.transform import FundamentalMatrixTransform
from skimage.measure import ransac


class MatchFeatures(pr.Processor):
    def __init__(self, matcher=BruteForceMatcher(k=2), ratio_test=True):
        super(MatchFeatures, self).__init__()
        self.matcher = matcher
        self.ratio_test = ratio_test

    def call(self,  des1, des2):
        matches = self.matcher(des1, des2)
        if self.ratio_test:
            matches = match_ratio_test(matches, ratio=0.6)
        # points = get_match_points(kps1, kps2, matches)
        # indices = get_match_indices(matches)
        # return points, indices
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
        image1, image2 = images[:2]

        # extract features
        # kps1, des1 = detect_ORB_fratures(image1)
        # kps2, des2 = detect_ORB_fratures(image2)

        kps1, des1 = detect_SIFT_features(image1)
        kps2, des2 = detect_SIFT_features(image2)

        # match features
        matches = brute_force_matcher(des1, des2)
        matches = match_ratio_test(matches, ratio=0.75)
        points1, points2 = get_match_points(kps1, kps2, matches)
        indices = get_match_indices(matches)


        # skimage ransac
        model, inliers = ransac((points1, points2),
                                # EssentialMatrixTransform, min_samples=8,
                                FundamentalMatrixTransform, min_samples=8,
                                residual_threshold=0.5, max_trials=1000)

        matches = np.array(matches)[inliers]
        p1_inliers = points1[inliers]
        p2_inliers = points2[inliers]
        print('Number of inliers', p1_inliers.shape[0])

        # compute fundamental matrix
        # F = model.params
        F, _ = self.compute_fundamental_matrix(p1_inliers, p2_inliers)

        # compute_essential_matrix
        E = self.compute_essential_matrix(F)

        # recover camera pose
        points, rotation, translation, mask = self.recover_pose(
            E, p1_inliers, p2_inliers)

        # get projection matrix
        P1 = self.K @ self.projection_matrix
        P2 = rotation @ self.projection_matrix
        P2[:3, 3] = translation.ravel()
        P2 = self.K @ P2

        # triangulation
        points3D = self.triangulate_points(P1, P2, p1_inliers.T, p2_inliers.T)
        print(points3D.shape)

        kps2 = np.array(kps2)
        des2 = np.array(des2)
        i = np.array(indices[1])[inliers]

        base_kps = kps2[i]
        base_des = des2[i]
        print(i.shape)
        print('good till here')
        return self.warp(points3D, [base_kps, base_des], P2)


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
        base_kps, base_des = inference['base_features']
        P2 = inference['P2']
        # points, indices = self.find_correspondences(base_features, images[3])

        for arg in range(len(images)-2):

            # keypoints1, descriptor1 = self.detector(images[arg + 1])
            # keypoints2, descriptor2 = self.detector(images[arg + 2])

            keypoints1, descriptor1 = detect_SIFT_features(images[arg + 1])
            keypoints2, descriptor2 = detect_SIFT_features(images[arg + 2])

            matches = brute_force_matcher(base_des, descriptor2)
            matches = match_ratio_test(matches, ratio=0.75)
            indices = get_match_indices(matches)
            points1, points2 = get_match_points(base_kps, keypoints2, matches)

            model, inliers = ransac((points1, points2),
                                    FundamentalMatrixTransform, min_samples=8,
                                    residual_threshold=2.0, max_trials=100)

            matches = np.array(matches)[inliers]
            p1_inliers = points1[inliers]
            p2_inliers = points2[inliers]
            i = np.array(indices[0])[inliers]
            print('Number of inliers****', p1_inliers.shape[0])

            _, rotation, translation = self.solve_pnp(points3d[i],
                                                      p1_inliers)
            rotation = self.rotation_vector_to_matrix(rotation)
            P3 = rotation @ self.projection_matrix
            P3[:3, 3] = translation.ravel()
            P3 = self.K @ P3

            matches = brute_force_matcher(descriptor1, descriptor2)
            matches = match_ratio_test(matches, ratio=0.75)
            indices = get_match_indices(matches)

            points1, points2 = get_match_points(keypoints1, keypoints2,
                                                matches)

            model, inliers = ransac((points1, points2),
                                    FundamentalMatrixTransform, min_samples=8,
                                    residual_threshold=0.5, max_trials=1000)

            p1_inliers = points1[inliers]
            p2_inliers = points2[inliers]
            points3d = self.triangulate_points(P2, P3, p1_inliers.T,
                                               p2_inliers.T)

            print(points3d.shape)
            self.points3D.append(points3d)
            P2 = P3.copy()

            keypoints2 = np.array(keypoints2)
            descriptor2 = np.array(descriptor2)
            i = np.array(indices[1])[inliers]

            base_kps = keypoints2[i]
            base_des = descriptor2[i]
            print(i.shape)
            print('good till here agin')

            print('##################')
            # break
        plot_3D_keypoints(self.points3D)
        return self.points3D

# todo
# RANSAC implemenatation in numpy 
# eight point algorithm implementation in numpy.
