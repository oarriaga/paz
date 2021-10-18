from tensorflow.keras.losses import Loss
import tensorflow as tf


class WeightedForeground(Loss):
    def __init__(self, beta=3.0):
        super(WeightedForeground, self).__init__()
        self.beta = beta

    def _extract_alpha_mask(self, RGBA_mask):
        alpha_mask = RGBA_mask[:, :, :, 3:4]
        color_mask = RGBA_mask[:, :, :, 0:3]
        return color_mask, alpha_mask

    def call(self, RGBA_mask_true, RGB_mask_pred):
        RGB_mask_true, alpha_mask = self._extract_alpha_mask(RGBA_mask_true)
        foreground_true = RGB_mask_true * alpha_mask
        foreground_pred = RGB_mask_pred * alpha_mask
        background_true = RGB_mask_true * (1.0 - alpha_mask)
        background_pred = RGB_mask_true * (1.0 - alpha_mask)
        foreground_loss = tf.abs(foreground_true - foreground_pred)
        background_loss = tf.abs(background_true - background_pred)
        loss = (self.beta * foreground_loss) + background_loss
        loss = tf.reduce_mean(loss, axis=[1, 2, 3])
        loss = tf.math.minimum(loss, tf.float32.max)
        return loss
