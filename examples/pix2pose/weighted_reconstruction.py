from tensorflow.keras.losses import Loss
import tensorflow as tf


def split_alpha_mask(RGBA_mask):
    """Splits alpha mask and RGB image.

    # Arguments
        RGBA_mask: Tensor [batch, H, W, 4]

    # Returns
        Color tensor [batch, H, W, 3] and alpha tensor [batch, H, W, 1]
    """
    color_mask = RGBA_mask[:, :, :, 0:3]
    alpha_mask = RGBA_mask[:, :, :, 3:4]
    return color_mask, alpha_mask


def split_error_mask(RGBE_mask):
    """Splits error mask and RGB image.

    # Arguments
        RGBA_mask: Tensor [batch, H, W, 4]

    # Returns
        Color tensor [batch, H, W, 3] and error tensor [batch, H, W, 1]

    """
    color_mask = RGBE_mask[:, :, :, 0:3]
    error_mask = RGBE_mask[:, :, :, 3:4]
    return color_mask, error_mask


def compute_foreground_loss(RGB_true, RGB_pred, alpha_mask):
    """Computes foreground reconstruction L1 loss by using only positive alpha
        mask values.

    # Arguments
        RGB_true: Tensor [batch, H, W, 3]. True RGB label values.
        RGB_pred: Tensor [batch, H, W, 3]. Predicted RGB values.
        alpha_mask: Tensor [batch, H, W, 1]. True normalized alpha mask values.

    # Returns
        Tensor [batch, H, W, 3] with foreground loss values.
    """
    foreground_true = RGB_true * alpha_mask
    foreground_pred = RGB_pred * alpha_mask
    foreground_loss = tf.abs(foreground_true - foreground_pred)
    return foreground_loss


def compute_background_loss(RGB_true, RGB_pred, alpha_mask):
    """Computes the L1 reconstruction loss, weighting the inverted alpha
        mask values in the predicted RGB image by beta.

    # Arguments
        RGB_true: Tensor [batch, H, W, 3]. True RGB label values.
        RGB_pred: Tensor [batch, H, W, 3]. Predicted RGB values.
        alpha_mask: Tensor [batch, H, W, 1]. True normalized alpha mask values.

    # Returns
        Tensor [batch, H, W, 3] with background loss values.
    """
    background_true = RGB_true * (1.0 - alpha_mask)
    background_pred = RGB_pred * (1.0 - alpha_mask)
    background_loss = tf.abs(background_true - background_pred)
    return background_loss


def compute_weighted_reconstruction_loss(RGBA_true, RGB_pred, beta=3.0):
    """Computes the L1 reconstruction loss, weighting the positive alpha
        mask values in the predicted RGB image by beta.

    # Arguments
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGB_pred: Tensor [batch, H, W, 3]. Predicted RGB values.
        beta: Float. Value used to multiple positive alpha mask values.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.

    """
    RGB_true, alpha_mask = split_alpha_mask(RGBA_true)
    foreground_loss = compute_foreground_loss(RGB_true, RGB_pred, alpha_mask)
    background_loss = compute_background_loss(RGB_true, RGB_pred, alpha_mask)
    reconstruction_loss = (beta * foreground_loss) + background_loss
    return tf.reduce_mean(reconstruction_loss, axis=-1, keepdims=True)


def normalized_image_to_normalized_device_coordinates(image):
    """Map image value from [0, 1] -> [-1, 1].
    """
    return (image * 2.0) - 1.0


def normalized_device_coordinates_to_normalized_image(image):
    """Map image value from [0, 1] -> [-1, 1].
    """
    return (image + 1.0) / 2.0


def compute_weighted_reconstruction_loss_with_error(
        RGBA_true, RGBE_pred, beta=3.0):
    """Computes L1 reconstruction loss by multiplying positive alpha mask
        by beta.

    # Arguments
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGBE_pred: Tensor [batch, H, W, 4]. Predicted RGB and error mask.
        beta: Float. Value used to multiple positive alpha mask values.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.

    """
    RGB_pred, error_mask = split_error_mask(RGBE_pred)
    loss = compute_weighted_reconstruction_loss(RGBA_true, RGB_pred, beta)
    return loss


class WeightedReconstruction(Loss):
    """Computes L1 reconstruction loss by multiplying positive alpha mask
        by beta.

    # Arguments
        beta: Float. Value used to multiple positive alpha mask values.
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGB_pred: Tensor [batch, H, W, 3]. Predicted RGB values.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.

    """
    def __init__(self, beta=3.0):
        super(WeightedReconstruction, self).__init__()
        self.beta = beta

    def call(self, RGBA_true, RGB_pred):
        loss = compute_weighted_reconstruction_loss(
            RGBA_true, RGB_pred, self.beta)
        return loss


class WeightedReconstructionWithError(Loss):
    """Computes L1 reconstruction loss by multiplying positive alpha mask
        by beta.

    # Arguments
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGBE_pred: Tensor [batch, H, W, 4]. Predicted RGB and error mask.
        beta: Float. Value used to multiple positive alpha mask values.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.

    """
    def __init__(self, beta=3.0):
        super(WeightedReconstructionWithError, self).__init__()
        self.beta = beta

    def call(self, RGBA_true, RGBE_pred):
        reconstruction_loss = compute_weighted_reconstruction_loss_with_error(
            RGBA_true, RGBE_pred, self.beta)
        return reconstruction_loss
