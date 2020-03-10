import numpy as np
from scipy import spatial
from ..core import Processor


class PoseEvaluation(Processor):
    """Abstract class for calculating pose metrics
    """
    def __init__(self, processor):
        self.processor = processor


class TranslationalError(Processor):
    """Computes translational error

    # Arguments
        topic: String. Topic to retrieve translation vector
        y_pred: predected labels
        y_true: true lables
    """
    def __init__(self, topic='translation'):
        super(TranslationalError, self).__init__()
        self.topic = topic

    def __call__(self, y_pred, y_true):
        ground_truth_translation = Processor(y_true)[self.topic]
        predicted_translation = Processor(y_pred)[self.topic]
        differnce = ground_truth_translation - predicted_translation
        translation_error = np.linalg.norm(differnce)
        return translation_error


class RotationalError(Processor):
    """Computes rotational error between two rotation matrices

    #Arguments
        topic: String. Topic to retrieve rotation matrix
        y_pred: predicted lables
        y_true: true labels
    """
    def __init__(self, topic='rotation'):
        super(RotationalError, self).__init__()
        self.topic = topic

    def __call__(self, y_pred, y_true):
        ground_truth_rotation = Processor(y_true)[self.topic]
        predicted_rotation = Processor(y_pred)[self.topic]
        distance_rotations = predicted_rotation.dot(ground_truth_rotation.T)
        rotational_error = np.arccos((np.trace(distance_rotations) - 1) / 2)
        return rotational_error


class ADD(Processor):
    """Computes Average distance between distinguishable views

    #Arguments
        topic: Topic to retrieve rotations and translations
        points3D: 3D model points
        predictions: predicted poses
        ground_truth: ground truth poses
    """
    def __init__(self, points3D, topic=None):
        super(ADD, self).__init__()
        self.topic = topic
        self.points3D = points3D

    def __call__(self, predictions, ground_truth):
        rotations_pred, translations_pred, _ = predictions
        translations_pred = translations_pred.reshape((3, 1))
        pred_transforms = rotations_pred.dot(self.points3D) + translations_pred

        rotations_true, translations_true, _ = ground_truth
        translations_true = translations_true.reshape((3, 1))
        true_transforms = rotations_true.dot(self.points3D) + translations_true

        add_norm = np.linalg.norm(pred_transforms - true_transforms, axis=1)
        ADD_error = add_norm.mean()
        return ADD_error

    def transform_poses(self, rotations, translations):
        translations = translations.reshape((3, 1))
        transformed_poses = rotations.dot(self.points3D) + translations
        return transformed_poses.T


class ADI(Processor):
    """Computes Average distance between indistinguishable views

    #Arguments
        topic: Topic to retrieve rotations and translations
        points3D: 3D model points
        predictions: predicted poses
        ground_truth: ground truth poses
    """
    def __init__(self, points3D, topic=None):
        super(ADI, self).__init__()
        self.topic = topic
        self.points3D = points3D

    def __call__(self, predictions, ground_truth):
        rotations_pred, translations_pred, _ = predictions
        translations_pred = translations_pred.reshape((3, 1))
        pred_transforms = rotations_pred.dot(self.points3D) + translations_pred

        rotations_true, translations_true, _ = ground_truth
        translations_true = translations_true.reshape((3, 1))
        true_transforms = rotations_true.dot(self.points3D) + translations_true

        kd_tree = spatial.cKDTree(pred_transforms)
        kd_tree_distances, ids = kd_tree.query(true_transforms, k=1)
        ADI_error = kd_tree_distances.mean()
        return ADI_error

    def transform_poses(self, rotations, translations):
        translations = translations.reshape((3, 1))
        transformed_poses = rotations.dot(self.points3D) + translations
        return transformed_poses.T

