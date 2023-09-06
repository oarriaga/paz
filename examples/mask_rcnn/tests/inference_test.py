import numpy as np
import pytest
import cv2

from mask_rcnn import inference


@pytest.fixture
def test_results():
    file_path = 'television.jpeg'
    image = cv2.imread(file_path)
    weights_path = 'mask_rcnn_coco.h5'
    return inference.test(image, weights_path, 128, 81, 1,
                          1, (32, 64, 128, 256, 512), [1024, 1024], 1)[0]


@pytest.mark.parametrize('box', [np.array([[295., 871., 361., 996.],
                                           [265., 716., 659., 915.],
                                           [224., 916., 432., 954.],
                                           [335.,  36., 593., 212.],
                                           [304., 255., 584., 467.],
                                           [305.,  67., 358., 167.],
                                           [250., 101., 414., 132.],
                                           [91., 217., 295., 753.],
                                           [399., 349., 544., 810.],
                                           [277., 501., 670., 667.],
                                           [330., 915., 384., 957.],
                                           [380.,   4., 526., 553.],
                                           [190.,  89., 682., 161.],
                                           [534., 714., 682., 1002.]])])
def test_bounding_box(test_results, box):
    boxes = np.array(test_results['rois'])
    np.testing.assert_array_equal(box, boxes)


@pytest.mark.parametrize('mask_shape', [(682, 1023)])
def test_mask_shape(test_results, mask_shape):
    num_obj = (test_results['masks']).shape[2]
    for i in range(num_obj):
        masks = test_results['masks']
        masks = np.array(masks)[:, :, i]
        assert (mask_shape == masks.shape)


@pytest.mark.parametrize('ones', [(3739, 33163, 3609, 20467, 29334, 2884, 2437,
                                   89748, 22181, 22762, 1796, 33690, 8159,
                                   27570)])
def test_mask(test_results, ones):
    num_obj = (test_results['masks'].shape)[2]
    masks = test_results['masks']
    for i in range(num_obj):
        mask = masks[:, :, i]
        mask = np.array(mask)
        assert(ones[i] == np.sum(mask))


@pytest.mark.parametrize('classes', [(59, 1,  59, 1,  1,  59, 59, 63,  1, 1,
                                      76, 1, 1, 58)])
def test_class_id(test_results, classes):
    ids = test_results['class_ids']
    ids = np.array(ids)
    for i in range(len(ids)):
        assert(classes[i] == ids[i])
