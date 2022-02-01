from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf


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
        RGB_true_rotated = tf.einsum('ij,bklj->bkli', rotation, RGB_true)
        RGB_true_rotated = normalized_device_coordinates_to_normalized_image(
            RGB_true_rotated)
        RGB_true_rotated = tf.clip_by_value(RGB_true_rotated, 0.0, 1.0)
        RGB_true_rotated = RGB_true_rotated * alpha
        RGBA_true_rotated = tf.concat([RGB_true_rotated, alpha], axis=3)
        loss = compute_weighted_reconstruction_loss(
            RGBA_true_rotated, RGB_pred, beta)
        loss = tf.expand_dims(loss, -1)
        symmetric_losses.append(loss)
    symmetric_losses = tf.concat(symmetric_losses, axis=-1)
    minimum_symmetric_loss = tf.reduce_min(symmetric_losses, axis=-1)
    return minimum_symmetric_loss


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
    # TODO check we need to set minimum to 1.0?
    error_true = tf.minimum(error_true, 1.0)
    error_loss = mean_squared_error(error_true, error_pred)
    error_loss = tf.expand_dims(error_loss, axis=-1)
    return error_loss


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


from paz.backend.image import draw_dot


def draw_points2D_(image, keypoints, colors, radius=1):
    for (u, v), (R, G, B) in zip(keypoints, colors):
        color = (int(R), int(G), int(B))
        draw_dot(image, (u, v), color, radius)
    return image


def rotate_image(image, rotation_matrix):
    """Rotates an image with a symmetry.

    # Arguments
        image: Array (H, W, 3) with domain [0, 255].
        rotation_matrix: Array (3, 3).

    # Returns
        Array (H, W, 3) with domain [0, 255]
    """
    mask_image = np.sum(image, axis=-1, keepdims=True) != 0
    image = image_to_normalized_device_coordinates(image)
    rotated_image = np.einsum('ij,klj->kli', rotation_matrix, image)
    rotated_image = normalized_device_coordinates_to_image(rotated_image)
    rotated_image = np.clip(rotated_image, a_min=0.0, a_max=255.0)
    rotated_image = rotated_image * mask_image
    return rotated_image
