import numpy as np
from scipy import spatial
from ..core.ops import numpy_ops as ops


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
            pose6D_true = self.processor(true_sample)[self.topic]
            true_translation = pose6D_true.translation
            pose6D_pred = self.processor(pred_sample)[self.topic]
            pred_translation = pose6D_pred.translation
        return np.linalg.norm(true_translation - pred_translation)


class QuaternionError(PoseEvaluation):
    def __init__(self, processor, topic='pose6D'):
        super(QuaternionError, self).__init__(processor, topic)

    def __call__(self, y_true, y_pred):
        pose6D_true = self.processor(y_true)[self.topic]
        true_quaternions = pose6D_true.quaternion

        pose6D_pred = self.processor(y_pred)[self.topic]
        pred_quaternions = pose6D_pred.quaternion
        distance_quaternions = pred_quaternions.dot(true_quaternions.T)
        quaternion_error = 2 * np.arccos(distance_quaternions.real)
        return np.rad2deg(quaternion_error)


class ADD(PoseEvaluation):
    def __init__(self, processor, topic='pose6D'):
        super(ADD, self).__init__(processor, topic)

    def __call__(self, points3D, y_true, y_pred):
        pose6D_true = self.processor(y_true)[self.topic]
        true_quaternions = pose6D_true.quaternion
        true_translations = pose6D_true.translation
        true_rotations = ops.quaternion_to_rotation_matrix(true_quaternions)
        true_translations = true_translations.reshape((3, 1))
        true_transforms = true_rotations.dot(points3D) + true_translations

        pose6D_pred = self.processor(y_pred)[self.topic]
        pred_quaternions = pose6D_pred.quaternion
        pred_translations = pose6D_pred.translation
        pred_rotations = ops.quaternion_to_rotation_matrix(pred_quaternions)
        pred_translations = pred_translations.reshape((3, 1))
        pred_transforms = pred_rotations.dot(points3D) + pred_translations

        aDD_norm = np.linalg.norm(pred_transforms - true_transforms, axis=1)
        aDD_error = aDD_norm.mean()
        return aDD_error


class ADI(PoseEvaluation):
    def __init__(self, processor, topic='pose6D'):
        super(ADI, self).__init__(processor, topic)

    def __call__(self, points3D, y_true, y_pred):
        pose6D_true = self.processor(y_true)[self.topic]
        true_quaternions = pose6D_true.quaternion
        true_translations = pose6D_true.translation
        true_rotations = ops.quaternion_to_rotation_matrix(true_quaternions)
        true_translations = true_translations.reshape((3, 1))
        true_transforms = true_rotations.dot(points3D) + true_translations

        pose6D_pred = self.processor(y_pred)[self.topic]
        pred_quaternions = pose6D_pred.quaternion
        pred_translations = pose6D_pred.translation
        pred_rotations = ops.quaternion_to_rotation_matrix(pred_quaternions)
        pred_translations = pred_translations.reshape((3, 1))
        pred_transforms = pred_rotations.dot(points3D) + pred_translations

        kd_tree = spatial.cKDTree(pred_transforms)
        kd_tree_distances, ids = kd_tree.query(true_transforms, k=1)
        aDI_error = kd_tree_distances.mean()
        return aDI_error
