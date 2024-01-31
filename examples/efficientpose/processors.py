import numpy as np
from paz.abstract import Processor


class RegressTranslation(Processor):
    """Applies regression offset values to translation anchors
    to get the 2D translation center-point and Tz.

    # Arguments
        translation_priors: Array of shape `(num_boxes, 3)`,
            translation anchors.

    # References
        This module is derived based on [EfficientPose](
            https://github.com/ybkscht/EfficientPose)
    """
    def __init__(self, translation_priors):
        self.translation_priors = translation_priors
        super(RegressTranslation, self).__init__()

    def call(self, translation_raw):
        return regress_translation(translation_raw, self.translation_priors)


def regress_translation(translation_raw, translation_priors):
    """Applies regression offset values to translation anchors
    to get the 2D translation center-point and Tz.

    # Arguments
        translation_raw: Array of shape `(1, num_boxes, 3)`,
        translation_priors: Array of shape `(num_boxes, 3)`,
            translation anchors.

    # Returns
        Array: of shape `(num_boxes, 3)`.

    # References
        This module is derived based on [EfficientPose](
            https://github.com/ybkscht/EfficientPose)
    """
    stride = translation_priors[:, -1]
    x = translation_priors[:, 0] + (translation_raw[:, :, 0] * stride)
    y = translation_priors[:, 1] + (translation_raw[:, :, 1] * stride)
    Tz = translation_raw[:, :, 2]
    return np.concatenate((x, y, Tz), axis=0).T


class ComputeTxTyTz(Processor):
    """Computes the Tx and Ty components of the translation vector
    with a given 2D-point and the intrinsic camera parameters.

    # References
        This module is derived based on [EfficientPose](
            https://github.com/ybkscht/EfficientPose)
    """
    def __init__(self, translation_scale_norm=1000):
        self.translation_scale_norm = translation_scale_norm
        super(ComputeTxTyTz, self).__init__()

    def call(self, translation_xy_Tz, camera_matrix, image_scale):
        return compute_tx_ty_tz(translation_xy_Tz, camera_matrix,
                                image_scale, self.translation_scale_norm)


def compute_tx_ty_tz(translation_xy_Tz, camera_matrix, image_scale,
                     translation_scale_norm=1000):
    """Computes Tx, Ty and Tz components of the translation vector
    with a given 2D-point and the intrinsic camera parameters.

    # Arguments
        translation_xy_Tz: Array of shape `(num_boxes, 3)`,
        camera_matrix: Array of shape `(3, 3)` camera matrix.
        translation_scale_norm: Float, factor to change units.
            EfficientPose internally works with meter and if the
            dataset unit is mm for example, then this parameter
            should be set to 1000.

    # Returns
        Array: of shape `(num_boxes, 3)`.

    # References
        This module is derived based on [EfficientPose](
            https://github.com/ybkscht/EfficientPose)
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    px, py = camera_matrix[0, 2], camera_matrix[1, 2]

    x = translation_xy_Tz[:, 0] / image_scale
    y = translation_xy_Tz[:, 1] / image_scale
    tz = translation_xy_Tz[:, 2] * translation_scale_norm

    x = x - px
    y = y - py
    tx = np.multiply(x, tz) / fx
    ty = np.multiply(y, tz) / fy
    tx, ty, tz = tx[np.newaxis, :], ty[np.newaxis, :], tz[np.newaxis, :]
    return np.concatenate((tx, ty, tz), axis=0).T
