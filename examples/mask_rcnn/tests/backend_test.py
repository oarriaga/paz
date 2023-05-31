import pytest
import tensorflow as tf
import numpy as np

from mask_rcnn.backend.image import subtract_mean_image, add_mean_image
from mask_rcnn.backend.image import generate_smaller_masks, generate_original_masks

from mask_rcnn.backend.boxes import compute_rpn_match, compute_anchor_boxes_overlaps
from mask_rcnn.backend.boxes import norm_boxes, denorm_boxes
from mask_rcnn.backend.boxes import generate_anchors, extract_boxes_from_mask
from mask_rcnn.backend.boxes import compute_rpn_bounding_box, normalize_log_refinement


@pytest.fixture
def boxes():
    boxes = tf.constant([[[149, 75, 225, 147],
                        [217, 137, 295, 220],
                        [214, 110, 243, 159],
                        [180, 179, 211, 205]]], dtype=tf.float32)
    return boxes


@pytest.fixture
def image():
    image = tf.constant([[[[[0., 1., 0.],
                         [0., 1., 1.]],
                        [[1., 0., 1.],
                         [0., 1., 0.]]]]], dtype=tf.float32)
    return image


@pytest.fixture
def mask():
    mask = tf.constant([[0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.]], dtype=tf.float32)
    return mask


@pytest.fixture
def data():
    size = (320, 320)
    shapes = Shapes(1, size)
    dataset = shapes.load_data()
    image = dataset[0]['image']
    mask = dataset[0]['masks']
    box_data = dataset[0]['box_data']
    data = [{'image': image, 'boxes': box_data, 'mask': mask}]
    return data


@pytest.fixture
def anchors():
    config = ShapesConfig()
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = generate_pyramid_anchors((8, 16, 32, 64, 128),
                                       [0.5, 1, 2], backbone_shapes,
                                       [4, 8, 16, 32, 64], 1)
    return anchors


@pytest.fixture
def anchor_boxes_overlaps(data, anchors):
    gt_class_ids = data[0]['boxes'][:, 4]
    gt_boxes = data[0]['boxes'][:, :4]
    over_laps, crowd_bool = compute_anchor_boxes_overlaps(anchors, gt_class_ids, gt_boxes)
    return over_laps, crowd_bool


@pytest.fixture
def rpn_match(anchor_boxes_overlaps, anchors):
    over_laps, crowd_bool = anchor_boxes_overlaps
    rpn, anchor_iou_max = compute_rpn_match(anchors, over_laps, crowd_bool)
    return rpn, anchor_iou_max


@pytest.fixture
def rpn_box(rpn_match, data, anchors, config):
    rpn, anchor_iou_max = rpn_match
    gt_boxes = data[0]['boxes'][:, :4]
    rpn_box = compute_rpn_bbox(gt_boxes, rpn, anchors, config, anchor_iou_max)
    return rpn_box


def test_generate_anchors():
    scales = [32, 64, 128]
    ratios = [0.5, 1, 2]
    shape = [256, 256]
    feature_stride = 32
    anchor_stride = 32
    boxes = generate_anchors(scales, ratios, shape, feature_stride, anchor_stride)
    assert boxes.shape == (576, 4)


def test_compute_anchor_boxes_overlaps(anchors, anchor_boxes_overlaps):
    overlaps, no_crowd_bool = anchor_boxes_overlaps
    assert anchors.shape[0] == overlaps.shape[0] == no_crowd_bool.shape[0]


def test_rpn_match(rpn_match):
    rpn, anchor_iou_max = rpn_match
    assert np.all(np.unique(rpn) == (-1, 0, 1))
    assert np.all(np.unique(anchor_iou_max) == (0, 1, 2))


def test_compute_rpn_box(rpn_box):
    box = rpn_box
    assert box.shape[0] == 256


@pytest.mark.parametrize('subtract_mean', [[[-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]],
                                           [[0.5, -0.5, 0.5], [-0.5, 0.5, -0.5]]])
def test_subtract_mean_image(image, subtract_mean):
    mean_pixel = [0.5, 0.5, 0.5]
    mean_val = subtract_mean_image(image, mean_pixel)
    assert mean_val == subtract_mean


@pytest.mark.parametrize('add_mean', [[[0.5, 1.5, 0.5], [0.5, 1.5, 1.5]],
                                      [[1.5, 0.5, 1.5], [0.5, 1.5, 0.5]]])
def test_add_mean_image(image, add_mean):
    mean_pixel = [0.5, 0.5, 0.5]
    mean_val = subtract_mean_image(image, mean_pixel)
    assert mean_val == add_mean


@pytest.mark.parametrize('small_mask', [[1., 1., 1., 1.], [1., 1., 1., 1.]])
def test_small_mask(mask, small_mask):
    boxes = [[1, 1, 2, 2]]
    reduced_mask = generate_smaller_masks(boxes, mask, (2, 4))
    assert reduced_mask == small_mask


@pytest.mark.parametrize('norm_boxes', [[1.1732284, 0.5905512, 1.7637795,  1.1496063],
                                        [1.7086614, 1.0787401, 2.3149607, 1.7244095],
                                        [1.6850394, 0.86614174, 1.9055119, 1.2440945],
                                        [1.4173229, 1.4094489, 1.6535434, 1.6062992]])
def test_original_mask(boxes, norm_boxes):
    boxes_normalised = norm_boxes(boxes, [128, 128])
    assert boxes_normalised == norm_boxes


@pytest.mark.parametrize('norm_boxes', [[1.1732284, 0.5905512, 1.7637795,  1.1496063],
                                        [1.7086614, 1.0787401, 2.3149607, 1.7244095],
                                        [1.6850394, 0.86614174, 1.9055119, 1.2440945],
                                        [1.4173229, 1.4094489, 1.6535434, 1.6062992]])
def test_original_mask(boxes, norm_boxes):
    boxes_denormalised = denorm_boxes(norm_boxes, [128, 128])
    assert boxes_denormalised == boxes


@pytest.mark.parametrize('boxes', [[5, 10, 20, 30]])
def test_apply_box_refinement(boxes):
    ground_box = [10, 20, 30, 40]
    refinement = normalize_log_refinement(boxes, ground_box)
    assert refinement == [0.5, 0.5, 0.28768207245178085, 0.0]


@pytest.mark.parametrize('bbox', [[1, 1, 2, 2]])
def test_small_mask(mask, bbox):
    box = extract_boxes_from_mask(mask)
    assert box == bbox
