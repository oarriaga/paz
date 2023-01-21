import os
import pytest
import numpy as np

from tensorflow.keras.utils import get_file
from paz import processors as pr
from paz.backend.image import load_image


@pytest.fixture
def boxes_with_label():
    box_with_label = np.array([[47., 239., 194., 370., 12.],
                               [7., 11., 351., 497., 15.],
                               [138., 199., 206., 300., 19.],
                               [122., 154., 214., 194., 18.],
                               [238., 155., 306., 204., 9.]])
    return box_with_label


def test_expand_pass_by_reference(boxes_with_label):
    initial_boxes_with_label = boxes_with_label.copy()
    expand = pr.Expand(probability=1.0)
    expand(np.ones((300, 300, 3)), boxes_with_label)
    assert np.all(initial_boxes_with_label == boxes_with_label)


def test_random_sample_crop_pass_by_reference(boxes_with_label):
    initial_boxes_with_label = boxes_with_label.copy()
    crop = pr.RandomSampleCrop(probability=1.0)
    crop(np.ones((300, 300, 3)), boxes_with_label)
    assert np.all(initial_boxes_with_label == boxes_with_label)


def test_random_sample_crop():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.9/object_detection_augmentation.png')
    filename = os.path.basename(URL)
    image_fullpath = get_file(filename, URL, cache_subdir='paz/tutorials')
    true_image = load_image(image_fullpath)
    H, W = true_image.shape[:2]
    true_boxes = np.array([[200 / W, 60 / H, 300 / W, 200 / H, 1],
                           [100 / W, 90 / H, 400 / W, 300 / H, 2]])

    class AugmentBoxes(pr.SequentialProcessor):
        def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
            super(AugmentBoxes, self).__init__()
            self.add(pr.ToImageBoxCoordinates())
            self.add(pr.RandomSampleCrop())
            self.add(pr.ToNormalizedBoxCoordinates())

    augment_boxes = AugmentBoxes()
    for _ in range(1000):
        crop_image, crop_boxes = augment_boxes(true_image, true_boxes)
        assert len(crop_boxes.shape) == 2
        assert np.alltrue(crop_boxes[:, 0] < crop_boxes[:, 2])
        assert np.alltrue(crop_boxes[:, 1] < crop_boxes[:, 3])
