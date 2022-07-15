import numpy as np
import pytest
import cv2

from mask_rcnn import inference


@pytest.fixture
def test_results():
    file_path = 'test_images/television.jpeg'
    image = cv2.imread(file_path)
    weights_path = 'weights/mask_rcnn_coco.h5'
    return inference.test([image], weights_path)[0]


@pytest.mark.parametrize('box', [np.array([[ 337,  661,  585,  969],
 [ 274,  894,  381,  974],
 [ 334,  232,  565,  488],
 [ 368,    1,  563,  252],
 [ 290,   84,  375,  149],
 [  69,  268,  316,  708],
 [ 364,  399,  577,  744],
 [ 333,  911,  380,  960],
 [ 546,  703,  682, 1023],
 [ 384,  147,  546,  302]])])
def test_bounding_box(test_results, box):
    boxes = np.array(test_results['rois'])
    np.testing.assert_array_equal(box,boxes)

@pytest.mark.parametrize('mask_shape', [(682, 1023)])
def test_mask_shape(test_results, mask_shape):
    num_obj = (test_results['masks']).shape[2]
    for i in range(num_obj):
        masks = test_results['masks']
        masks = np.array(masks)[:, :, i]
        assert (mask_shape == masks.shape)


@pytest.mark.parametrize('ones', [(37370, 5109, 29785, 22668, 3827, 103587, 28364, 1817, 28940, 14946, 1100, 28544)])
def test_mask(test_results, ones):
    num_obj = (test_results['masks'].shape)[2]
    masks = test_results['masks']
    mask=[]
    for i in range(num_obj):
        mask = masks[:,:,i]
        mask = np.array(mask)
        assert(ones[i] == np.sum(mask))

@pytest.mark.parametrize('classes', [(1,  59, 1,  1,  59, 63,  1,  76, 58,  1, 76, 58)])
def test_class_id(test_results, classes):
   ids = test_results['class_ids']
   ids = np.array(ids)
   for i in range (len(ids)) :
         assert(classes[i] ==ids[i])
