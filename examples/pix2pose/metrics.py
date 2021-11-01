from loss import compute_weighted_reconstruction_loss_with_error
from loss import compute_error_prediction_loss
from loss import compute_weighted_reconstruction_loss
import tensorflow as tf


def error_prediction(RGBA_true, RGBE_pred, beta=3.0):
    return compute_error_prediction_loss(RGBA_true, RGBE_pred)


def mean_squared_error(y_true, y_pred):
    squared_difference = tf.square(y_true[:, :, :, 0:3] - y_pred[:, :, :, 0:3])
    return tf.reduce_mean(squared_difference, axis=-1)


def weighted_reconstruction_wrapper(beta=3.0, with_error=False):
    if with_error:
        def weighted_reconstruction(y_true, y_pred):
            return compute_weighted_reconstruction_loss_with_error(
                y_true, y_pred, beta)
    else:
        def weighted_reconstruction(y_true, y_pred):
            return compute_weighted_reconstruction_loss(y_true, y_pred, beta)
    return weighted_reconstruction
