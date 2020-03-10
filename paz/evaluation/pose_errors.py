import numpy as np
from scipy import spatial
from scipy.spatial.transform import Rotation


class PoseEvaluation(object):
    """Evaluates the predicted poses against ground truth
    poses

    #Arguments
        topic: Topic to retrieve pose6D messages
        processor: instance of object Processor
    """
    def __init__(self, processor, topic=None):
        self.processor = processor
        self.topic = topic


class TranslationError(PoseEvaluation):
    def __init__(self, processor, topic='pose6D'):
        super(TranslationError, self).__init__(processor, topic)

    def __call__(self, y_true, y_pred):
        for true_sample, pred_sample in zip(y_true, y_pred):
            true_translation = self.processor(true_sample)[self.topic][1]
            pred_translation = self.processor(pred_sample)[self.topic][1]
        return np.linalg.norm(true_translation - pred_translation)


class RotationalError(PoseEvaluation):
    def __init__(self, processor, topic='pose6D'):
        super(RotationalError, self).__init__(processor, topic)

    def __call__(self, y_true, y_pred):
        true_quaternions = self.processor(y_true)[self.topic][0]
        true_rotations = Rotation.as_matrix(true_quaternions)
        pred_quaternions = self.processor(y_pred)[self.topic][0]
        pred_rotations = Rotation.as_matrix(pred_quaternions)
        distance_rotations = pred_rotations.dot(true_rotations.T)
        rotational_error = np.arccos((np.trace(distance_rotations) - 1) / 2)
        return np.rad2deg(rotational_error)


class ADD(PoseEvaluation):
    def __init__(self, processor, topic='pose6D'):
        super(ADD, self).__init__(processor, topic)

    def __call__(self, points3D, y_true, y_pred):
        ground_truth = self.processor(y_true)[self.topic]
        true_quaternions, true_translations, _ = ground_truth
        true_rotations = Rotation.as_matrix(true_quaternions)
        true_translations = true_translations.reshape((3, 1))
        true_transforms = true_rotations.dot(points3D) + true_translations

        predictions = self.processor(y_pred)[self.topic]
        pred_quaternions, pred_translations, _ = predictions
        pred_rotations = Rotation.as_matrix(pred_quaternions)
        pred_translations = pred_translations.reshape((3, 1))
        pred_transforms = pred_rotations.dot(points3D) + pred_translations

        aDD_norm = np.linalg.norm(pred_transforms - true_transforms, axis=1)
        aDD_error = aDD_norm.mean()
        return aDD_error


class ADI(PoseEvaluation):
    def __init__(self, processor, topic='pose6D'):
        super(ADI, self).__init__(processor, topic)

    def __call__(self, points3D, y_true, y_pred):
        ground_truth = self.processor(y_true)[self.topic]
        true_quaternions, true_translations, _ = ground_truth
        true_rotations = Rotation.as_matrix(true_quaternions)
        true_translations = true_translations.reshape((3, 1))
        true_transforms = true_rotations.dot(points3D) + true_translations

        predictions = self.processor(y_pred)[self.topic]
        pred_quaternions, pred_translations, _ = predictions
        pred_rotations = Rotation.as_matrix(pred_quaternions)
        pred_translations = pred_translations.reshape((3, 1))
        pred_transforms = pred_rotations.dot(points3D) + pred_translations

        kd_tree = spatial.cKDTree(pred_transforms)
        kd_tree_distances, ids = kd_tree.query(true_transforms, k=1)
        ADI_error = kd_tree_distances.mean()
        return ADI_error
