import numpy as np
import pytest
import cv2

from mask_rcnn import inference

@pytest.fixture
def image():
    file_path = '/home/incendio/Desktop/test_images/elephant.jpg'
    return cv2.imread(file_path)


@pytest.fixture
def predicted_box():
    return np.array([34, 55, 359, 592])


def test_inference(image, predicted_box):
    boxes = inference.test(image)[0]['rois']
    assert (boxes[0].all() == predicted_box.all())
