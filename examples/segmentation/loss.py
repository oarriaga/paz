import tensorflow as tf
from tensorflow.keras.losses import Loss


def compute_F_beta_score(y_true, y_pred, beta=0.0, class_weights=1.0):
    true_positives = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    false_positives = tf.reduce_sum(y_pred, axis=[1, 2]) - true_positives
    false_negatives = tf.reduce_sum(y_true, axis=[1, 2]) - true_positives
    numerator = (1 + beta(**2)) * true_positives
    denominator = numerator + ((beta**2) * false_negatives) + false_positives
    F_beta_score = numerator / (denominator + 1e-5)
    return class_weights * F_beta_score


def compute_jaccard_score(y_true, y_pred, class_weights=1.0):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2]) - intersection
    jaccard_score = (intersection) / (union + 1e-5)
    return class_weights * jaccard_score


def compute_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, 1e-5, 1.0 - 1e-5)
    modulator = alpha * tf.pow(1 - y_pred, gamma)
    focal_loss = - modulator * y_true * tf.log(y_pred)
    return focal_loss


class DiceLoss(Loss):
    def __init__(self, beta=0.0, class_weights=1.0):
        self.beta = beta
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        args = (self.beta, self.class_weights)
        return compute_F_beta_score(y_true, y_pred, *args)


class JaccardScore(Loss):
    def __init__(self, class_weights):
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        return compute_jaccard_score(y_true, y_pred, self.class_weights)


class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return compute_focal_loss(y_true, y_pred, self.gamma, self.alpha)
