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


class ClipBoxes(Processor):
    def __init__(self, shape):
        self.shape = shape
        super(ClipBoxes, self).__init__()

    def call(self, boxes):
        return clip_boxes(boxes, self.shape)


def clip_boxes(boxes, shape):
    H, W = shape
    decoded_boxes = boxes[:, :4]
    boxes[:, 0] = np.clip(decoded_boxes[:, 0], 0, W - 1)
    boxes[:, 1] = np.clip(decoded_boxes[:, 1], 0, H - 1)
    boxes[:, 2] = np.clip(decoded_boxes[:, 2], 0, W - 1)
    boxes[:, 3] = np.clip(decoded_boxes[:, 3], 0, H - 1)
    return boxes
