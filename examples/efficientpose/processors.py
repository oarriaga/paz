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
    def __init__(self):
        super(ComputeTxTyTz, self).__init__()

    def call(self, translation_xy_Tz, camera_parameter):
        return compute_tx_ty_tz(translation_xy_Tz, camera_parameter)


def compute_tx_ty_tz(translation_xy_Tz, camera_parameter):
    """Computes Tx, Ty and Tz components of the translation vector
    with a given 2D-point and the intrinsic camera parameters.

    # Arguments
        translation_xy_Tz: Array of shape `(num_boxes, 3)`,
        camera_parameter: Array: of shape `(6,)` camera parameter.

    # Returns
        Array: of shape `(num_boxes, 3)`.

    # References
        This module is derived based on [EfficientPose](
            https://github.com/ybkscht/EfficientPose)
    """
    fx, fy = camera_parameter[0], camera_parameter[1],
    px, py = camera_parameter[2], camera_parameter[3],
    tz_scale, image_scale = camera_parameter[4], camera_parameter[5]

    x = translation_xy_Tz[:, 0] / image_scale
    y = translation_xy_Tz[:, 1] / image_scale
    tz = translation_xy_Tz[:, 2] * tz_scale

    x = x - px
    y = y - py
    tx = np.multiply(x, tz) / fx
    ty = np.multiply(y, tz) / fy
    tx, ty, tz = tx[np.newaxis, :], ty[np.newaxis, :], tz[np.newaxis, :]
    return np.concatenate((tx, ty, tz), axis=0).T
