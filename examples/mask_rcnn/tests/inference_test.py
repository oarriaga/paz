import numpy as np
import pytest
import cv2

from mask_rcnn import inference

@pytest.fixture
def test_results():
    file_path = '/home/incendio/Documents/Thesis/test_images/elephant.jpg'
    image = cv2.imread(file_path)
    return inference.test(image)[0]

@pytest.mark.parametrize('box', [np.array([34, 55, 359, 592])])
def test_bounding_box(test_results, box):
    boxes = test_results['rois']
    assert (boxes[0].all() == box.all())


@pytest.mark.parametrize('mask_shape', [(426, 640)])
def test_mask_shape(test_results, mask_shape):
    masks = test_results['masks']
    masks = np.array(masks)[:, :, 0]
    assert (mask_shape == masks.shape)


@pytest.mark.parametrize('ones', [79848])
def test_mask(test_results, ones):
    masks = test_results['masks']
    masks = np.array(masks)[:, :, 0]
    assert(ones == np.sum(masks))
