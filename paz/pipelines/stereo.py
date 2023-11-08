import numpy as np
from paz import processors as pr
from paz.backend.stereo import get_match_indices
from paz.backend.stereo import contruct_projection_matrix
from paz.backend.stereo import triangulate_points_np
from paz.backend.stereo import extract_keypoints_RGB


class FindCorrespondances(pr.Processor):
    def __init__(self, match_ratio, residual_thresh):
        super(FindCorrespondances, self).__init__()
        self.detector = pr.DetectSIFTFeatures()
        self.match_features = pr.MatchFeatures(match_ratio=match_ratio)
        self.get_inliers = pr.ExtractMatchInliersPoints(
            residual_thresh=residual_thresh)

    def call(self, base_feature, image):
        base_keypoints, base_descriptors = base_feature
        keypoints1, descriptor1 = self.detector(image)
        matches = self.match_features(base_descriptors, descriptor1)
        points1, points2, inliers = self.get_inliers(
            matches, base_keypoints, keypoints1)
        indices = get_match_indices(matches)
        base_indices_inliers = indices[0][inliers]
        return points1, points2, base_indices_inliers


class EstimateDepthFromTwoView(pr.Processor):
    def __init__(self, camera_intrinsics, optimizer=None,
                 bundle_adjustment=False, residual_thresh=0.5,
                 match_ratio=0.75):
        super(EstimateDepthFromTwoView, self).__init__()
        self.K = camera_intrinsics
        self.optimizer = optimizer
        self.bundle_adjustment = bundle_adjustment
        self.detector = pr.DetectSIFTFeatures()
        self.match_features = pr.MatchFeatures(match_ratio=match_ratio)
        self.get_fundamental_matrix = pr.ComputeFundamentalMatrixRANSAC(
            residual_thresh=residual_thresh)
        self.get_inliers = pr.ExtractMatchInliersPoints(
            residual_thresh=residual_thresh)
        self.local_bundal_adjustment = pr.LocalBundleAdjustment(self.optimizer,
                                                                self.K)
        self.compute_essential_matrix = pr.ComputeEssentialMatrix(self.K)
        self.recover_pose = pr.RecoverPose(self.K)
        self.warp = pr.WrapOutput(['points3D', 'base_feature', 'P2', 'colors'])

    def call(self, image1, image2):
        keypoints1, descriptors1 = self.detector(image1)
        keypoints2, descriptors2 = self.detector(image2)
        matches = self.match_features(descriptors1, descriptors2)
        inlier_points = self.get_inliers(matches, keypoints1, keypoints2)
        points1, points2, inliers = inlier_points
        colors = extract_keypoints_RGB(image1, points1)

        fundamental_matrix, _ = self.get_fundamental_matrix(points1, points2)
        essential_matrix = self.compute_essential_matrix(fundamental_matrix)
        R, t = self.recover_pose(essential_matrix, points1, points2)

        P1 = self.K @ np.eye(3, 4)
        P2 = contruct_projection_matrix(R, t)
        P2 = self.K @ P2

        points3D = triangulate_points_np(P1, P2, points1, points2)
        if self.bundle_adjustment:
            points3D, _ = self.local_bundal_adjustment(R, t, points3D, points2)

        indices1, indices2 = get_match_indices(matches)
        base_keypoints = keypoints2[indices2[inliers]]
        base_descriptors = descriptors2[indices2[inliers]]
        base_feature = (base_keypoints, base_descriptors)
        return self.warp(np.array(points3D), base_feature, P2, colors)


class StructureFromMotion(pr.Processor):
    def __init__(self, camera_intrinsics, optimizer=None,
                 bundle_adjustment=False, match_ratio=0.75,
                 residual_thresh=0.5, correspondence_thresh=0.5):
        super(StructureFromMotion, self).__init__()
        self.K = camera_intrinsics
        self.optimizer = optimizer
        self.bundle_adjustment = bundle_adjustment
        self.initialize_sfm = EstimateDepthFromTwoView(
            camera_intrinsics, optimizer, bundle_adjustment, residual_thresh,
            match_ratio)
        self.detector = pr.DetectSIFTFeatures()
        self.match_features = pr.MatchFeatures(match_ratio=match_ratio)
        self.get_inliers = pr.ExtractMatchInliersPoints(
            residual_thresh=residual_thresh)
        self.find_correspondences = FindCorrespondances(match_ratio,
                                                        correspondence_thresh)
        self.solve_pnp = pr.SolvePnP(camera_intrinsics)
        self.rotation_vector_to_matrix = pr.RotationVectorToRotationMatrix()
        self.local_bundle_adjustment = pr.LocalBundleAdjustment(self.optimizer,
                                                                self.K)
        self.points3D = []
        self.colors = []
        self.warp = pr.WrapOutput(['points3D', 'colors'])

    def call(self, images):
        print(f'Processing image 1/{len(images)-1}')
        inference = self.initialize_sfm(images[0], images[1])
        points3D = inference['points3D']
        self.points3D.append(points3D)
        self.colors.append(inference['colors'])
        base_feature = inference['base_feature']
        P2 = inference['P2']

        for arg in range(len(images)-2):
            print(f'Processing image {arg + 2}/{len(images)-1}')
            image1 = images[arg + 1]
            image2 = images[arg + 2]
            keypoints1, descriptor1 = self.detector(images[arg + 1])
            keypoints2, descriptor2 = self.detector(images[arg + 2])

            correspondences = self.find_correspondences(base_feature, image2)
            points1, points2, indices = correspondences
            points3D = points3D[indices]
            _, R, t = self.solve_pnp(points3D, points1)
            R = self.rotation_vector_to_matrix(R)
            P3 = contruct_projection_matrix(R, t)
            P3 = self.K @ P3

            matches = self.match_features(descriptor1, descriptor2)
            inlier_points = self.get_inliers(matches, keypoints1, keypoints2)
            points1, points2, inliers = inlier_points
            colors = extract_keypoints_RGB(image1, points1)
            points3D = triangulate_points_np(P2, P3, points1, points2)
            if self.bundle_adjustment:
                points3D, _ = self.local_bundle_adjustment(R, t, points3D,
                                                           points2)

            P2 = P3.copy()
            self.points3D.append(points3D)
            self.colors.append(colors)
            indices = get_match_indices(matches)
            arg = indices[1][inliers]
            base_feature = [keypoints2[arg], descriptor2[arg]]
        return self.warp(self.points3D, self.colors)
