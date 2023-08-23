import cv2
import numpy as np
from paz.abstract import Processor


class ComputeResizingShape(Processor):
    def __init__(self, size):
        self.size = size
        super(ComputeResizingShape, self).__init__()

    def call(self, image):
        return compute_resizing_shape(image, self.size)


def compute_resizing_shape(image, size):
    H, W = image.shape[:2]
    scale = size / max(H, W)
    resizing_W = int(W * scale)
    resizing_H = int(H * scale)    
    resizing_shape = (resizing_W, resizing_H)
    return resizing_shape, scale


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
