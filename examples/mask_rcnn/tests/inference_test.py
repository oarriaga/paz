import numpy as np
import pytest
import cv2

from mask_rcnn import inference

@pytest.fixture
def image():
    file_path = '/home/incendio/Documents/Thesis/test_images/elephant.jpg'
    return cv2.imread(file_path)

@pytest.mark.parametrize('box', [np.array([34, 55, 359, 592])])
def test_bounding_box(image, box):
    boxes = inference.test(image)[0]['rois']
    assert (boxes[0].all() == box.all())


@pytest.mark.parametrize('mask_shape', [(426, 640)])
def test_mask_shape(image, mask_shape):
    masks = inference.test(image)[0]['masks']
    masks = np.array(masks)[:, :, 0]
    assert (mask_shape == masks.shape)


@pytest.mark.parametrize('ones', [79848])
def test_mask(image, ones):
    masks = inference.test(image)[0]['masks']
    masks = np.array(masks)[:, :, 0]
    assert(ones == np.sum(masks))
