import tensorflow as tf
from tensorflow.keras.losses import Loss


def compute_jaccard_score(y_true, y_pred, class_weights=1.0):
    """Computes the Jaccard score. The Jaccard score is the intersection
    over union of the predicted with respect to real masks.

    # Arguments
        y_true: Tensor of shape ``(batch, H, W, num_channels)``.
        y_pred: Tensor of shape ``(batch, H, W, num_channels)``.
        class_weights: Float or list of floats of shape ``(num_classes)``.

    # Returns
        Tensor of shape ``(batch)`` containing the F beta score per sample.
    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2]) - intersection
    jaccard_score = (intersection) / (union + 1e-5)
    return class_weights * jaccard_score


class JaccardLoss(Loss):
    """Computes the Jaccard loss. The Jaccard score is the intersection
    over union of the predicted with respect to real masks.

    # Arguments
        class_weights: Float or list of floats of shape ``(num_classes)``.
    """
    def __init__(self, class_weights=1.0):
        super(JaccardLoss, self).__init__()
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        return 1.0 - compute_jaccard_score(y_true, y_pred, self.class_weights)

