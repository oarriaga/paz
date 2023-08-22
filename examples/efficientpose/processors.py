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
