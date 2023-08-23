import cv2
import numpy as np
from paz.abstract import Processor
from paz.backend.boxes import to_corner_form

class ComputeResizingShape(Processor):
    def __init__(self, size):
        self.size = size
        super(ComputeResizingShape, self).__init__()

    def call(self, image):
        return compute_resizing_shape(image, self.size)


def compute_resizing_shape(image, size):
    H, W = image.shape[:2]
    image_scale = size / max(H, W)
    resizing_W = int(W * image_scale)
    resizing_H = int(H * image_scale)
    resizing_shape = (resizing_W, resizing_H)
    return resizing_shape, image_scale


class PadImage(Processor):
    def __init__(self, size, mode='constant'):
        self.size = size
        self.mode = mode
        super(PadImage, self).__init__()

    def call(self, image):
        return pad_image(image, self.size, self.mode)


def pad_image(image, size, mode):
    H, W = image.shape[:2]
    pad_H = size - H
    pad_W = size - W
    image = np.pad(image, [(0, pad_H), (0, pad_W), (0, 0)], mode=mode)
    return image


class ComputeCameraParameter(Processor):
    def __init__(self, camera_matrix, translation_scale_norm):
        self.camera_matrix = camera_matrix
        self.translation_scale_norm = translation_scale_norm
        super(ComputeCameraParameter, self).__init__()

    def call(self, image_scale):
        return compute_camera_parameter(image_scale, self.camera_matrix,
                                        self.translation_scale_norm)


def compute_camera_parameter(image_scale, camera_matrix,
                             translation_scale_norm):
    camera_parameter = np.array([camera_matrix[0, 0],
                                 camera_matrix[1, 1],
                                 camera_matrix[0, 2],
                                 camera_matrix[1, 2],
                                 translation_scale_norm,
                                 image_scale])
    return camera_parameter


class DecodeBoxes(Processor):
    """Decodes bounding boxes.

    # Arguments
        prior_boxes: Numpy array of shape (num_boxes, 4).
        variances: List of two float values.
    """
    def __init__(self, prior_boxes, variances=[0.1, 0.1, 0.2, 0.2]):
        self.prior_boxes = prior_boxes
        self.variances = variances
        super(DecodeBoxes, self).__init__()

    def call(self, boxes):
        decoded_boxes = decode(boxes, self.prior_boxes, self.variances)
        return decoded_boxes


def decode(predictions, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Decode default boxes into the ground truth boxes

    # Arguments
        loc: Numpy array of shape `(num_priors, 4)`.
        priors: Numpy array of shape `(num_priors, 4)`.
        variances: List of two floats. Variances of prior boxes.

    # Returns
        decoded boxes: Numpy array of shape `(num_priors, 4)`.
    """
    center_x = predictions[:, 0:1] * priors[:, 2:3] * variances[0]
    center_x = center_x + priors[:, 0:1]
    center_y = predictions[:, 1:2] * priors[:, 3:4] * variances[1]
    center_y = center_y + priors[:, 1:2]
    W = priors[:, 2:3] * np.exp(predictions[:, 2:3] * variances[2])
    H = priors[:, 3:4] * np.exp(predictions[:, 3:4] * variances[3])
    boxes = np.concatenate([center_x, center_y, W, H], axis=1)
    boxes = to_corner_form(boxes)
    return np.concatenate([boxes, predictions[:, 4:]], 1)
