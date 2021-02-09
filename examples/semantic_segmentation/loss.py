import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss


def compute_F_beta_score(y_true, y_pred, beta=1.0, class_weights=1.0):
    true_positives = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    false_positives = tf.reduce_sum(y_pred, axis=[1, 2]) - true_positives
    false_negatives = tf.reduce_sum(y_true, axis=[1, 2]) - true_positives
    B_squared = tf.math.pow(beta, 2)
    numerator = (1.0 + B_squared) * true_positives
    denominator = numerator + (B_squared * false_negatives) + false_positives
    F_beta_score = numerator / (denominator + 1e-5)
    return class_weights * F_beta_score


def compute_jaccard_score(y_true, y_pred, class_weights=1.0):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2]) - intersection
    jaccard_score = (intersection) / (union + 1e-5)
    return class_weights * jaccard_score


def compute_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, 1e-5, 1.0 - 1e-5)
    modulator = alpha * tf.math.pow(1 - y_pred, gamma)
    focal_loss = - modulator * y_true * tf.math.log(y_pred)
    return focal_loss


class DiceLoss(Loss):
    def __init__(self, beta=1.0, class_weights=1.0):
        super(DiceLoss, self).__init__()
        self.beta = beta
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        args = (self.beta, self.class_weights)
        return 1.0 - compute_F_beta_score(y_true, y_pred, *args)


class JaccardLoss(Loss):
    def __init__(self, class_weights=1.0):
        super(JaccardLoss, self).__init__()
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        return 1.0 - compute_jaccard_score(y_true, y_pred, self.class_weights)


class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return compute_focal_loss(y_true, y_pred, self.gamma, self.alpha)


def test_jaccard_score_full_overlap():
    y_true = np.ones((32, 128, 128, 1))
    y_pred = np.ones((32, 128, 128, 1))
    score = compute_jaccard_score(y_true, y_pred).numpy()
    assert np.allclose(1.0, score)


def test_jaccard_score_no_overlap():
    y_true = np.ones((32, 128, 128, 1))
    y_pred = np.zeros((32, 128, 128, 1))
    score = compute_jaccard_score(y_true, y_pred).numpy()
    assert np.allclose(0.0, score)


def test_jaccard_score_half_overlap():
    y_true = np.ones((32, 128, 128, 1))
    y_pred_A = np.ones((16, 128, 128, 1))
    y_pred_B = np.zeros((16, 128, 128, 1))
    y_pred = np.concatenate([y_pred_A, y_pred_B], axis=0)
    score = compute_jaccard_score(y_true, y_pred).numpy()
    assert np.allclose(0.5, np.mean(score))


def test_jaccard_loss_full_overlap():
    y_true = np.ones((32, 128, 128, 1))
    y_pred = np.ones((32, 128, 128, 1))
    score = JaccardLoss()(y_true, y_pred).numpy()
    assert np.allclose(0.0, score)


def test_jaccard_loss_no_overlap():
    y_true = np.ones((32, 128, 128, 1))
    y_pred = np.zeros((32, 128, 128, 1))
    score = JaccardLoss()(y_true, y_pred).numpy()
    assert np.allclose(1.0, score)


def test_jaccard_loss_half_overlap():
    y_true = np.ones((32, 128, 128, 1))
    y_pred_A = np.ones((16, 128, 128, 1))
    y_pred_B = np.zeros((16, 128, 128, 1))
    y_pred = np.concatenate([y_pred_A, y_pred_B], axis=0)
    score = JaccardLoss()(y_true, y_pred).numpy()
    assert np.allclose(0.5, np.mean(score))


test_jaccard_score_no_overlap()
test_jaccard_score_half_overlap()
test_jaccard_score_full_overlap()
test_jaccard_loss_no_overlap()
test_jaccard_loss_half_overlap()
test_jaccard_loss_full_overlap()
