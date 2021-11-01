from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf


def extract_alpha_mask(RGBA_mask):
    color_mask = RGBA_mask[:, :, :, 0:3]
    alpha_mask = RGBA_mask[:, :, :, 3:4]
    return color_mask, alpha_mask


def extract_error_mask(RGBE_mask):
    color_mask = RGBE_mask[:, :, :, 0:3]
    error_mask = RGBE_mask[:, :, :, 3:4]
    return color_mask, error_mask


def compute_foreground_loss(RGB_true, RGB_pred, alpha_mask):
    foreground_true = RGB_true * alpha_mask
    foreground_pred = RGB_pred * alpha_mask
    foreground_loss = tf.abs(foreground_true - foreground_pred)
    return foreground_loss


def compute_background_loss(RGB_true, RGB_pred, alpha_mask):
    background_true = RGB_true * (1.0 - alpha_mask)
    background_pred = RGB_pred * (1.0 - alpha_mask)
    background_loss = tf.abs(background_true - background_pred)
    return background_loss


def compute_weighted_reconstruction_loss(RGBA_true, RGB_pred, beta=3.0):
    RGB_true, alpha_mask = extract_alpha_mask(RGBA_true)
    foreground_loss = compute_foreground_loss(RGB_true, RGB_pred, alpha_mask)
    background_loss = compute_background_loss(RGB_true, RGB_pred, alpha_mask)
    reconstruction_loss = (beta * foreground_loss) + background_loss
    return tf.reduce_mean(reconstruction_loss, axis=-1, keepdims=True)


def compute_weighted_reconstruction_loss_with_error(RGBA_true, RGBE_pred,
                                                    beta=3.0):
    RGB_pred, error_mask = extract_error_mask(RGBE_pred)
    loss = compute_weighted_reconstruction_loss(RGBA_true, RGB_pred, beta)
    return loss


def compute_error_prediction_loss(RGBA_true, RGBE_pred):
    RGB_pred, error_pred = extract_error_mask(RGBE_pred)
    error_true = compute_weighted_reconstruction_loss(RGBA_true, RGB_pred, 1.0)
    error_true = tf.minimum(error_true, 1.0)
    error_loss = mean_squared_error(error_true, error_pred)
    error_loss = tf.expand_dims(error_loss, axis=-1)
    return error_loss


class WeightedReconstructionWithError(Loss):
    def __init__(self, beta=3.0):
        super(WeightedReconstructionWithError, self).__init__()
        self.beta = beta

    def call(self, RGBA_true, RGBE_pred):
        reconstruction = compute_weighted_reconstruction_loss_with_error(
            RGBA_true, RGBE_pred, self.beta)
        error_prediction = compute_error_prediction_loss(RGBA_true, RGBE_pred)
        loss = reconstruction + error_prediction
        return loss


class WeightedReconstruction(Loss):
    def __init__(self, beta=3.0):
        super(WeightedReconstruction, self).__init__()
        self.beta = beta

    def call(self, RGBA_true, RGB_pred):
        loss = compute_weighted_reconstruction_loss(
            RGBA_true, RGB_pred, self.beta)
        return loss


def MSE_without_last_channel(y_true, y_pred):
    squared_difference = tf.square(y_true[:, :, :, 0:3] - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
