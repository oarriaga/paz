from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import mean_squared_error
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


def compute_weighted_symmetric_loss(RGBA_true, RGB_pred, rotations, beta=3.0):
    """Computes the mininum of all rotated L1 reconstruction losses weighting
        the positive alpha mask values in the predicted RGB image by beta.

    # Arguments
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGB_pred: Tensor [batch, H, W, 3]. Predicted RGB values.
        rotations: Array (num_symmetries, 3, 3). Rotation matrices
            that when applied lead to the same object view.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.
    """
    RGB_true, alpha = split_alpha_mask(RGBA_true)
    RGB_true = normalized_image_to_normalized_device_coordinates(RGB_true)
    symmetric_losses = []
    for rotation in rotations:
        RGB_true = tf.einsum('ij,bklj->bkli', rotation, RGB_true)
        RGB_true = normalized_device_coordinates_to_normalized_image(RGB_true)
        RGB_true = tf.concat([RGB_true, alpha], axis=3)
        loss = compute_weighted_reconstruction_loss(RGBA_true, RGB_pred, beta)
        loss = tf.expand_dims(loss, -1)
        symmetric_losses.append(loss)
    symmetric_losses = tf.concat(symmetric_losses, axis=-1)
    minimum_symmetric_loss = tf.reduce_min(symmetric_losses, axis=-1)
    return minimum_symmetric_loss


def compute_weighted_symmetric_loss2(RGBA_true, RGB_pred, rotations, beta=3.0):
    """Computes the mininum of all rotated L1 reconstruction losses weighting
        the positive alpha mask values in the predicted RGB image by beta.

    # Arguments
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGB_pred: Tensor [batch, H, W, 3]. Predicted RGB values.
        rotations: Array (num_symmetries, 3, 3). Rotation matrices
            that when applied lead to the same object view.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.
    """
    # alpha mask is invariant to rotations that leave the shape symmetric.
    RGB_true, alpha = split_alpha_mask(RGBA_true)
    # RGB_original_shape = tf.shape(RGBA_true)
    batch_size, H, W, num_channels = RGB_true.shape
    batch_size, H, W, num_channels = 32, 128, 128, 3
    RGB_true = tf.reshape(RGB_true, [batch_size, -1, 3])
    RGB_true = to_normalized_device_coordinates(RGB_true)
    RGB_pred = to_normalized_device_coordinates(RGB_pred)
    symmetric_losses = []
    for rotation in rotations:
        # RGB_true_symmetric = tf.matmul(rotation, RGB_true.T).T
        RGB_true_symmetric = tf.einsum('ij,klj->kli', rotation, RGB_true)
        RGB_true_symmetric = tf.reshape(RGB_true_symmetric, (batch_size, H, W, num_channels))
        RGBA_true_symmetric = tf.concat([RGB_true_symmetric, alpha], axis=3)
        symmetric_loss = compute_weighted_reconstruction_loss(
            RGBA_true_symmetric, RGB_pred, beta)
        symmetric_loss = tf.expand_dims(symmetric_loss, -1)
        symmetric_losses.append(symmetric_loss)
    symmetric_losses = tf.concat(symmetric_losses, axis=-1)
    minimum_symmetric_loss = tf.reduce_min(symmetric_losses, axis=-1)
    return minimum_symmetric_loss


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


def compute_error_prediction_loss(RGBA_true, RGBE_pred):
    """Computes L2 reconstruction loss of predicted error mask.

    # Arguments
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGBE_pred: Tensor [batch, H, W, 3]. Predicted RGB and error mask.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.

    """
    RGB_pred, error_pred = split_error_mask(RGBE_pred)
    error_true = compute_weighted_reconstruction_loss(RGBA_true, RGB_pred, 1.0)
    error_true = tf.minimum(error_true, 1.0)
    error_loss = mean_squared_error(error_true, error_pred)
    error_loss = tf.expand_dims(error_loss, axis=-1)
    return error_loss


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


class WeightedSymmetricReconstruction(Loss):
    """Computes the mininum of all rotated L1 reconstruction losses weighting
        the positive alpha mask values in the predicted RGB image by beta.
    """
    def __init__(self, rotations, beta=3.0):
        super(WeightedSymmetricReconstruction, self).__init__()
        self.rotations = rotations
        self.beta = beta

    def call(self, RGBA_true, RGB_pred):
        loss = compute_weighted_symmetric_loss(
            RGBA_true, RGB_pred, self.rotations, self.beta)
        return loss


class ErrorPrediction(Loss):
    """Computes L2 reconstruction loss of predicted error mask.

    # Arguments
        RGBA_true: Tensor [batch, H, W, 4]. Color with alpha mask label values.
        RGBE_pred: Tensor [batch, H, W, 3]. Predicted RGB and error mask.

    # Returns
        Tensor [batch, H, W] with weighted reconstruction loss values.

    """
    def __init__(self):
        super(ErrorPrediction, self).__init__()

    def call(self, RGBA_true, RGBE_pred):
        error_loss = compute_error_prediction_loss(RGBA_true, RGBE_pred)
        return error_loss


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
