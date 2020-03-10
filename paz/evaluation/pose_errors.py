import numpy as np
from scipy import spatial


class PoseError():
    """Computes the pose errors given the estimated and ground-truth poses

    # Arguments
    points3D: 3D model points
    R_est: Estimated rotation matrix of size 3x3
    t_est: Estimated translation vector of size 3x1
    R_gt: Ground-truth rotation matrix of size 3x3
    t_gt: Ground-truth translation vector of size 3x1

    # Return
    Computes different poses errors
    """
    def __init__(self, points3D, R_est, t_est, R_gt, t_gt):
        self.points3D = points3D
        self.rotation_est = R_est
        self.translation_est = t_est
        self.rotation_gt = R_gt
        self.translation_gt = t_gt

        self.transformed_pts_est = self.transform_3d(R_est, t_est)
        self.transformed_pts_gt = self.transform_3d(R_gt, t_gt)

    def transform_3d(self, rotations, translations):
        translations = translations.reshape((3, 1))
        transform_pts = rotations.dot(self.points3D) + translations
        return transform_pts.T

    def add(self):
        distance = np.linalg.norm(self.transformed_pts_est - self.transformed_pts_gt, axis=1)
        ADD_error = distance.mean()
        return ADD_error

    def adi(self):
        kd_tree = spatial.cKDTree(self.transformed_pts_est)
        kd_tree_distances, ids = kd_tree.query(self.transformed_pts_gt, k=1)
        adi_error = kd_tree_distances.mean()
        return adi_error

    def rotational_error(self):
        R_est = self.rotation_est
        R_gt_T = np.transpose(self.rotation_gt)
        rotational_error = np.rad2deg(np.arccos((np.trace(R_est.dot(R_gt_T)) - 1) / 2))
        return rotational_error

    def translational_error(self):
        translation_error = np.linalg.norm(self.translation_gt - self.translation_est)
        return translation_error


def translation_error(y_true, y_pred):
    ground_truth_translation = processor(y_true)
    predicted_translation = processor(y_pred)
    return np.linalg.norm(ground_truth_translation - predicted_translation)


class PoseEvaluation(object):
    def __init__(self, processor, topic='pose6D'):
        self.processor = processor


class TranslationError(PoseEvaluation):
    def __init__(self, processor, topic='translation'):
        super(TranslationError, self).__init__(processor, topic)

    def __call__(self, y_true, y_pred):
        for y_true_sample, y_pred_sample in zip(y_true, y_pred):
            ground_truth_translation = self.processor(y_true_sample)[self.topic]
            predicted_translation = self.processor(y_pred_sample)[self.topic]
        return np.linalg.norm(ground_truth_translation - predicted_translation)


class RotationalError(PoseEvaluation):
    def __init__(self, processor):
        super(TranslationError, self).__init__(processor)

    def __call__(self, y_true, y_pred):
        ground_truth_rotation = self.processor(y_true)
        predicted_rotation = self.processor(y_pred)
        R_gt_T = np.transpose(ground_truth_rotation)
        rotational_error = np.rad2deg(np.arccos((np.trace(R_est.dot(R_gt_T)) - 1) / 2))
        return rotational_error
