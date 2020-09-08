import pytest
import numpy as np

from paz import processors as pr


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
