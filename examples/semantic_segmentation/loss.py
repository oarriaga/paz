import tensorflow as tf
from tensorflow.keras.losses import Loss


def compute_F_beta_score(y_true, y_pred, beta=1.0, class_weights=1.0):
    """Computes the F beta score. The F beta score is the geometric mean
    of the precision and recall, where the recall is B times more important
    than the precision.

    # Arguments
        y_true: Tensor of shape ``(batch, H, W, num_channels)``.
        y_pred: Tensor of shape ``(batch, H, W, num_channels)``.
        beta: Float.
        class_weights: Float or list of floats of shape ``(num_classes)``.

    # Returns
        Tensor of shape ``(batch)`` containing the F beta score per sample.
    """
    true_positives = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    false_positives = tf.reduce_sum(y_pred, axis=[1, 2]) - true_positives
    false_negatives = tf.reduce_sum(y_true, axis=[1, 2]) - true_positives
    B_squared = tf.math.pow(beta, 2)
    numerator = (1.0 + B_squared) * true_positives
    denominator = numerator + (B_squared * false_negatives) + false_positives
    F_beta_score = numerator / (denominator + 1e-5)
    return class_weights * F_beta_score


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


class DiceLoss(Loss):
    """Computes the F beta loss. The F beta score is the geometric mean
    of the precision and recall, where the recall is B times more important
    than the precision.

    # Arguments
        beta: Float.
        class_weights: Float or list of floats of shape ``(num_classes)``.
    """
    def __init__(self, beta=1.0, class_weights=1.0):
        super(DiceLoss, self).__init__()
        self.beta = beta
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        args = (self.beta, self.class_weights)
        return 1.0 - compute_F_beta_score(y_true, y_pred, *args)


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
