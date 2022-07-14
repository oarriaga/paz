import tensorflow as tf
from tensorflow.keras.losses import Loss


def compute_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Computes the Focal loss. The Focal loss down weights
        properly classified examples.

    # Arguments
        y_true: Tensor of shape ``(batch, H, W, num_channels)``.
        y_pred: Tensor of shape ``(batch, H, W, num_channels)``.
        gamma: Float.
        alpha: Float.
        class_weights: Float or list of floats of shape ``(num_classes)``.

    # Returns
        Tensor of shape ``(batch)`` containing the F beta score per sample.
    """
    y_pred = tf.clip_by_value(y_pred, 1e-5, 1.0 - 1e-5)
    modulator = alpha * tf.math.pow(1 - y_pred, gamma)
    focal_loss = - modulator * y_true * tf.math.log(y_pred)
    return focal_loss


class FocalLoss(Loss):
    """Computes the Focal loss. The Focal loss down weights
        properly classified examples.

    # Arguments
        gamma: Float.
        alpha: Float.
        class_weights: Float or list of floats of shape ``(num_classes)``.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return compute_focal_loss(y_true, y_pred, self.gamma, self.alpha)
