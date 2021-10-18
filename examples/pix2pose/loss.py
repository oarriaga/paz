from tensorflow.keras.losses import Loss
import tensorflow as tf


class WeightedRGBMask(Loss):
    def __init__(self, beta=3.0, epsilon=1e-4):
        super(WeightedRGBMask, self).__init__()
        self.beta, self.epsilon = beta, epsilon

    def _extract_masks(RGBA_mask):
        # TODO this should be an additional input or extracted from alpha mask
        # mask_object = tf.math.ceil(RGB_mask)
        # mask_object = tf.math.reduce_max(mask_object, axis=-1, keepdims=True)
        # mask_object = tf.repeat(mask_object, repeats=3, axis=-1)
        # mask_background = tf.ones(tf.shape(mask_object)) - mask_object
        # return mask_object, mask_background
        return None

    def _extract_alpha_mask(self, RGBA_mask):
        alpha_mask = RGBA_mask[:, :, :, 3:4]
        color_mask = RGBA_mask[:, :, :, 0:3]
        return color_mask, alpha_mask

    def _compute_masks(self, alpha_mask):
        alpha_mask, 1.0 - alpha_mask

    def _unitball_to_normalized(x):
        # [-1, 1] -> [0, 1]
        return (x + 1) * 0.5

    def _normalized_to_unitball(x):
        # [0, 1] -> [-1, 1]
        return (2.0 * x) - 1.0

    def call(self, RGBA_mask_true, RGB_mask_pred):
        # Loss that penalizes more object color mismatch
        # Loss that penalizes less background color not being "0"
        # RGB_mask_true = self._unitball_to_normalized(RGB_mask_true)
        # mask_object, mask_background = self._extract_masks(RGB_mask_true)
        # RGB_mask_true = self._normalized_to_unitball(RGB_mask_true)
        # RGB_mask_true = RGB_mask_true + self.epsilon

        # Set the background to be all -1
        RGB_mask_true, alpha_mask = self._extract_alpha_mask(RGBA_mask_true)
        # object_mask, background_mask = self._compute_masks(alpha_mask)

        foreground_true = RGB_mask_true * alpha_mask
        foreground_pred = RGB_mask_pred * alpha_mask
        background_true = RGB_mask_true * (1.0 - alpha_mask)
        background_pred = RGB_mask_true * (1.0 - alpha_mask)
        foreground_loss = tf.abs(foreground_true - foreground_pred)
        background_loss = tf.abs(background_true - background_pred)
        loss = (self.beta * foreground_loss) + background_loss
        loss = tf.reduce_mean(loss, axis[1, 2, 3])
        # RGB_mask_true = RGB_mask_true * mask_object
        # RGB_mask_true = RGB_mask_true + (mask_background * tf.constant(-1.))

        # Calculate the difference between the real and predicted images including the mask
        # object_error = tf.abs(RGB_mask_pred * mask_object - RGB_mask_true * mask_object)
        # background_error = tf.abs(RGB_mask_pred * mask_background - RGB_mask_true * mask_background)

        object_error = tf.reduce_sum(object_error, axis=-1)
        background_error = tf.reduce_sum(background_error, axis=-1)

        loss = (self.beta * object_error) + background_error
        loss = tf.reduce_mean(loss, axis=[1, 2, 3])
        loss = tf.math.minimum(loss, tf.float32.max)
        return loss
