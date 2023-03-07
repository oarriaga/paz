import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
import numpy as np

from mask_rcnn.layers import ProposalLayer
from mask_rcnn.utils import norm_boxes_graph, fpn_classifier_graph
from mask_rcnn.model import MaskRCNN
from mask_rcnn.layer_utils import slice_batch, compute_ROI_level, apply_ROI_pooling, pad_ROIs_value
from mask_rcnn.layer_utils import rearrange_pooled_features, trim_anchors_by_score, refine_detections
from mask_rcnn.layer_utils import apply_box_delta, clip_image_boundaries, compute_delta_specific
from mask_rcnn.layer_utils import compute_NMS, compute_targets_from_groundtruth_values
from mask_rcnn.layer_utils import compute_keep, compute_refined_ROIs, compute_target_boxes
from mask_rcnn.layer_utils import filter_low_confidence, apply_NMS, get_top_detections, compute_IOU
from mask_rcnn.layer_utils import compute_ROI_overlaps, compute_target_masks, compute_target_class_ids
from mask_rcnn.layer_utils import refine_bbox, trim_zeros, apply_box_deltas, clip_boxes, compute_scaled_area
from mask_rcnn.layer_utils import transform_ROI_coordinates, compute_max_ROI_level, check_if_crowded

from tensorflow.python.framework.ops import enable_eager_execution, disable_eager_execution


# disable_eager_execution()


@pytest.fixture
def model():
    window = norm_boxes_graph((171, 0, 853, 1024), (640, 640))
    base_model = MaskRCNN(model_dir='../../mask_rcnn', image_shape=[128, 128, 3], backbone="resnet101",
                          batch_size=8, images_per_gpu=1,
                          rpn_anchor_scales=(32, 64, 128, 256, 512),
                          train_rois_per_image=200,
                          num_classes=81, window=window)
    return base_model


@pytest.fixture
def feature_maps(model):
    return model.keras_model.output


@pytest.fixture
def RPN_model(model, feature_maps):
    return model.RPN(feature_maps)


@pytest.fixture
def anchors():
    return Input(shape=[None, 4])


@pytest.fixture
def ground_truth(ground_truth_boxes):
    class_ids = Input(shape=[None], dtype=tf.int32)
    masks = Input(shape=[1024, 1024, None], dtype=bool)
    return class_ids, ground_truth_boxes, masks


@pytest.fixture
def ground_truth_boxes():
    input_image = Input(shape=[None, None, 3])
    input_boxes = Input(shape=[None, 4], dtype=tf.float32)
    boxes = Lambda(lambda x:
                   norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_boxes)
    return boxes


@pytest.fixture
def FPN_classifier(proposal_layer, feature_maps):
    _, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(proposal_layer,
                                                      feature_maps[:-1], 81,
                                                      [128, 128, 3])
    return mrcnn_class, mrcnn_bbox


@pytest.fixture
def proposal_layer(RPN_model, anchors):
    _, RPN_class, RPN_box = RPN_model
    return ProposalLayer(proposal_count=2000, nms_threshold=0.7, name='ROI',
                         rpn_bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
                         pre_nms_limit=6000,
                         images_per_gpu=1,
                         batch_size=1)([RPN_class, RPN_box, anchors])


@pytest.fixture
def proposal_layer_trim_by_score(RPN_model, anchors):
    _, RPN_class, RPN_box = RPN_model
    RPN_class = RPN_class[:, :, 1]
    RPN_box = RPN_box * np.reshape(np.array([0.1, 0.1, 0.2, 0.2]), [1, 1, 4])
    scores, deltas, pre_nms_anchors = trim_anchors_by_score(RPN_class, RPN_box,
                                                            anchors, 1, 6000)
    return scores, deltas, pre_nms_anchors


@pytest.fixture
def proposal_layer_apply_box_delta(proposal_layer_trim_by_score):
    scores, deltas, pre_nms_anchors = proposal_layer_trim_by_score
    boxes = apply_box_deltas(pre_nms_anchors, deltas, 1)
    boxes = clip_image_boundaries(boxes, 1)
    return boxes


@pytest.fixture
def proposal_layer_NMS(proposal_layer_apply_box_delta, proposal_layer_trim_by_score):
    boxes = proposal_layer_apply_box_delta
    scores, __, __ = proposal_layer_trim_by_score
    proposal_count = tf.repeat(2000, 1)
    threshold = tf.repeat(0.7, 1)

    proposals = slice_batch([boxes, scores, proposal_count, threshold], [], compute_NMS, 1)
    return proposals


@pytest.fixture
def detection_layer_batch(proposal_layer, FPN_classifier):
    rois = proposal_layer
    mrcnn_class, mrcnn_bbox = FPN_classifier
    detections_batch = slice_batch([rois, mrcnn_class, mrcnn_bbox],
                                   [tf.cast([0.1, 0.1, 0.2, 0.2], dtype=tf.float32),
                                    [0, 0, 128, 128], 0.7,
                                    100, tf.cast(0.3, dtype=tf.float32)],
                                   refine_detections, 1)
    return detections_batch


@pytest.fixture
def detection_layer_compute_delta_specific(FPN_classifier):
    mrcnn_class, mrcnn_bbox = FPN_classifier
    class_ids, class_scores, deltas_specific = slice_batch([mrcnn_class, mrcnn_bbox],
                                                           [], compute_delta_specific, 1)
    return class_ids, class_scores, deltas_specific


@pytest.fixture
def detection_layer_compute_refined_rois(proposal_layer,
                                         detection_layer_compute_delta_specific):
    rois = proposal_layer
    class_ids, class_scores, deltas_specific = detection_layer_compute_delta_specific
    refined_rois = slice_batch([rois, deltas_specific * np.array([0.1, 0.1, 0.2, 0.2])],
                               [[0., 0., 128., 128.]], compute_refined_ROIs, 1)
    return refined_rois


@pytest.fixture
def detection_layer_compute_keep(detection_layer_compute_refined_rois,
                                 detection_layer_compute_delta_specific):
    class_ids, class_scores, deltas_specific = detection_layer_compute_delta_specific
    refine_rois = detection_layer_compute_refined_rois
    keep = slice_batch([class_ids, class_scores, refine_rois], [0.7, 100, 0.3],
                       compute_keep, 1)
    return refine_rois, keep


@pytest.fixture
def detection_layer_filter_low_confidence(detection_layer_compute_delta_specific):
    class_ids, class_scores, deltas_specific = detection_layer_compute_delta_specific

    keep = tf.where(class_ids > 0)[:, 0]

    keep = slice_batch([class_scores], [keep, 0.7],
                       filter_low_confidence, 1)
    return keep


@pytest.fixture
def detection_layer_apply_nms(detection_layer_compute_delta_specific,
                              detection_layer_filter_low_confidence,
                              detection_layer_compute_refined_rois):
    class_ids, class_scores, deltas_specific = detection_layer_compute_delta_specific
    refined_rois = detection_layer_compute_refined_rois
    keep = detection_layer_filter_low_confidence

    nms_keep = slice_batch([class_ids, class_scores, refined_rois, keep],
                           [100, 0.3],
                           apply_NMS, 1)
    return nms_keep


@pytest.fixture
def detection_layer_get_top_detections(detection_layer_compute_delta_specific,
                                       detection_layer_filter_low_confidence,
                                       detection_layer_apply_nms):
    class_ids, class_scores, deltas_specific = detection_layer_compute_delta_specific
    nms_keep = detection_layer_apply_nms
    keep = detection_layer_filter_low_confidence

    keep = slice_batch([class_scores, keep, nms_keep], [100],
                       get_top_detections, 1)
    return keep


@pytest.fixture
def detection_target_batch(proposal_layer, ground_truth):
    ROIs = proposal_layer
    class_ids, boxes, masks = ground_truth
    names = ['rois', 'target_class_ids', 'target_bbox', 'target_mask']
    outputs = slice_batch([ROIs, class_ids, boxes, masks],
                          [256, 0.33, [28, 28], (56, 56),
                           tf.cast(np.array([0.1, 0.1, 0.2, 0.2]), dtype=tf.float32)],
                          compute_targets_from_groundtruth_values, 1, names=names)
    return outputs


@pytest.fixture
def detection_target_layer_detections_target(proposal_layer, ground_truth):
    class_ids, boxes, masks = ground_truth
    rois = proposal_layer
    rois, roi_class_ids, deltas, masks = slice_batch([rois, class_ids, boxes, masks],
                                                     [256, 0.33, [28, 28],
                                                      (56, 56), np.array([0.1, 0.1, 0.2, 0.2])],
                                                     compute_targets_from_groundtruth_values, 1)
    return rois, roi_class_ids, deltas, masks


@pytest.fixture
def detection_target_layer_compute_refined_boxes(proposal_layer, ground_truth):
    class_ids, boxes, masks = ground_truth
    rois = proposal_layer

    proposals, refined_class_ids, refined_boxes, refined_masks, crowd_boxes = \
        slice_batch([rois, class_ids, boxes, masks], [], check_if_crowded, 1)
    return proposals, refined_class_ids, refined_boxes, refined_masks, crowd_boxes


@pytest.fixture
def detection_target_layer_compute_IOU(proposal_layer, detection_target_layer_compute_refined_boxes):
    rois = proposal_layer
    __, __, refined_boxes, __, __ = detection_target_layer_compute_refined_boxes

    overlaps = slice_batch([rois, refined_boxes], [], compute_IOU, 1)
    return overlaps


@pytest.fixture
def detection_target_layer_compute_ROI_overlap(proposal_layer,
                                               detection_target_layer_compute_refined_boxes):
    rois = proposal_layer
    proposals, __, refined_boxes, __, crowd_boxes = detection_target_layer_compute_refined_boxes

    positive_overlaps, positive_rois, negative_rois = slice_batch([proposals, refined_boxes,
                                                                   crowd_boxes],
                                                                  [256, 0.33],
                                                                  compute_ROI_overlaps, 1)
    return positive_overlaps, positive_rois, negative_rois


@pytest.fixture
def detection_target_layer_compute_target(detection_target_layer_compute_refined_boxes,
                                          detection_target_layer_compute_ROI_overlap):
    proposals, refined_class_ids, refined_boxes, refined_masks, crowd_boxes = \
        detection_target_layer_compute_refined_boxes

    positive_overlaps, positive_ROIs, negative_ROIs = detection_target_layer_compute_ROI_overlap

    deltas, ROI_prior_boxes = \
        slice_batch([positive_overlaps, positive_ROIs,
                     refined_boxes], [np.array([0.1, 0.1, 0.2, 0.2])], compute_target_boxes, 1)

    roi_class_ids = \
        slice_batch([refined_class_ids, positive_overlaps], [], compute_target_class_ids, 1)

    masks = slice_batch([positive_ROIs, ROI_prior_boxes, refined_masks, positive_overlaps],
                        [[28, 28], (56, 56)],
                        compute_target_masks, 1)

    return deltas, roi_prior_boxes, roi_class_ids, masks


@pytest.fixture
def detection_target_pad_ROI(detection_target_layer_compute_target,
                             detection_target_layer_compute_ROI_overlap):
    positive_overlaps, positive_ROIs, negative_ROIs = detection_target_layer_compute_ROI_overlap
    deltas, ROI_prior_boxes, ROI_class_ids, masks = detection_target_layer_compute_target

    ROIs, ROI_class_ids, ROI_deltas, ROI_masks = slice_batch([positive_ROIs, negative_ROIs,
                                                              ROI_class_ids,
                                                              deltas, masks],
                                                             [256], pad_ROIs_value, 1)
    return ROIs, ROI_class_ids, ROI_deltas, ROI_masks


@pytest.fixture
def pyramid_ROI_level(proposal_layer, feature_maps):
    shape = (1024, 1024, 3)
    roi_level = compute_ROI_level(proposal_layer, shape)
    return roi_level


@pytest.fixture
def pyramid_ROI_pooling(proposal_layer, pyramid_ROI_level, feature_maps):
    roi_level = pyramid_ROI_level
    pooled, box_to_level = apply_ROI_pooling(roi_level, proposal_layer,
                                             feature_maps, [7, 7])
    return pooled, box_to_level


@pytest.fixture
def pyramid_ROI_pooled_features(proposal_layer, pyramid_ROI_pooling):
    pooled, box_to_level = pyramid_ROI_pooling
    box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
    box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                             axis=1)
    pooled = rearrange_pooled_features(pooled, box_to_level, proposal_layer)
    return pooled


def test_detection_layer_batch(detection_layer_batch):
    detections = detection_layer_batch
    assert detections.shape[0] == 1
    assert detections.shape[2] == 6


def test_detection_layer_fun(detection_layer_compute_keep,
                             detection_layer_filter_low_confidence,
                             detection_layer_apply_nms,
                             detection_layer_get_top_detections):
    refined_rois, __ = detection_layer_compute_keep
    result = detection_layer_filter_low_confidence
    nms_keep = detection_layer_apply_nms
    keep = detection_layer_get_top_detections

    assert refined_rois.shape == (1, None, 4)
    assert result.shape[0] == 1
    assert nms_keep.shape[0] == 1
    assert keep.shape[0] == 1


def test_proposal_layer_functions(proposal_layer_trim_by_score,
                                  proposal_layer_apply_box_delta):
    scores, deltas, pre_nms_anchors = proposal_layer_trim_by_score
    boxes = proposal_layer_apply_box_delta
    proposals = proposal_layer_NMS

    assert deltas.shape[2] == 4
    assert pre_nms_anchors.shape[2] == 4
    assert boxes.shape[2] == 4
    assert proposals


@pytest.mark.parametrize('shapes', [[(3,), (2,), (3,), (4,)]])
def test_detection_target_batch(detection_target_batch, shapes):
    ROIs, target_class, target_box, target_mask = detection_target_batch
    results_shape = [K.shape(ROIs).shape, K.shape(target_class).shape,
                     K.shape(target_box).shape, K.shape(target_mask).shape]
    assert results_shape == shapes


def test_detection_target_layer_functions(detection_target_layer_compute_refined_boxes,
                                          detection_target_layer_compute_IOU,
                                          detection_target_layer_compute_ROI_overlap,
                                          detection_target_layer_compute_target,
                                          detection_target_pad_ROI):
    proposals, refined_class_ids, refined_boxes, refined_masks, crowd_boxes = \
        detection_target_layer_compute_refined_boxes
    overlaps = detection_target_layer_compute_IOU
    positive_overlaps, positive_ROIs, negative_ROIs = detection_target_layer_compute_ROI_overlap

    deltas, ROI_prior_boxes, ROI_class_ids, masks = detection_target_layer_compute_target
    masks = detection_target_layer_target_masks
    ROIs, ROI_class_ids, ROI_deltas, ROI_masks = detection_target_pad_ROI

    assert ROIs.shape[2] == 4
    assert ROI_class_ids.shape[0] == 1
    assert deltas.shape[2] == 4
    assert (masks.shape[2], masks.shape[3]) == (28, 28)
    assert refined_boxes.shape[2] == 4
    assert overlaps.shape[0] == 1
    assert positive_ROIs.shape[2] == 4
    assert negative_ROIs.shape[2] == 4
    assert deltas.shape[2] == 4
    assert ROI_prior_boxes.shape[2] == 4
    assert (ROI_masks.shape[2], ROI_masks.shape[3], ROI_masks.shape[4]) == (1024, 1024, 1)


@pytest.mark.parametrize('shape', [(1024, 1024, 3)])
def test_pyramid_ROI_align_functions(pyramid_ROI_level, pyramid_ROI_pooling,
                                     pyramid_ROI_pooled_features, shape):
    roi_level = pyramid_ROI_level
    pooled, box_to_level = pyramid_ROI_pooling
    pooled = pyramid_ROI_pooled_features

    assert K.int_shape(roi_level) == (1, None)
    assert K.int_shape(box_to_level) == (None, 2)
    assert K.int_shape(pooled) == (1, None, 7, 7, 256)


@pytest.fixture
def test_results_refine_box():
    box = tf.Variable([[337, 661, 585, 969]])
    prior_box = tf.Variable([[350, 650, 590, 1000]])
    return refine_bbox(box, prior_box)


@pytest.mark.parametrize('delta_box', [[0.03629032, 0.03246753, -0.03278985, 0.12783337]])
def test_refine_box(test_results_refine_box, delta_box):
    prior_results = test_results_refine_box.numpy()
    np.testing.assert_almost_equal(prior_results[0], delta_box, decimal=6)


@pytest.fixture
def test_results_trim_zeros():
    box = tf.Variable([[0, 0, 0, 0], [0, 0, 0, 0],
                       [337, 661, 585, 969], [337, 661, 585, 969],
                       [337, 661, 585, 969], [337, 661, 585, 969],
                       [337, 661, 585, 969], [337, 661, 585, 969],
                       [0, 0, 0, 0], [0, 0, 0, 0]])
    return trim_zeros(box)


@pytest.mark.parametrize('boxes', [[[337, 661, 585, 969], [337, 661, 585, 969],
                                    [337, 661, 585, 969], [337, 661, 585, 969],
                                    [337, 661, 585, 969], [337, 661, 585, 969]]])
def test_trim_zeros(test_results_trim_zeros, boxes):
    box, non_zeros = test_results_trim_zeros
    np.testing.assert_almost_equal(box.numpy(), boxes)
    assert non_zeros.numpy().sum() == 6


@pytest.fixture
def test_results_apply_box_delta():
    box = tf.Variable([[337., 661., 500., 300.]])
    deltas = tf.Variable([[0.03629032, 0.03246753, -0.03278985, 0.12783337]])
    return apply_box_delta(box, deltas)


@pytest.mark.parametrize('result', [[345.54434, 673.8929, 503.28625, 263.66562]])
def test_apply_box_delta(test_results_apply_box_delta, result):
    values = test_results_apply_box_delta.numpy()
    np.testing.assert_almost_equal(values[0], result, decimal=4)


@pytest.fixture
def test_results_clip_boxes():
    box = tf.Variable([[337., 661., 700., 300.]])
    windows = np.array([400, 500, 800, 900], dtype=np.float32)
    return clip_boxes(box, windows)


@pytest.mark.parametrize('result', [[400., 661., 700., 500.]])
def test_clip_boxes(test_results_clip_boxes, result):
    values = test_results_clip_boxes.numpy()
    np.testing.assert_almost_equal(values[0], result, decimal=4)


@pytest.fixture
def test_results_transform_ROI_coordinates():
    box = tf.Variable([[337., 661., 700., 300.]])
    roi = tf.Variable([[300., 600., 800., 400.]])
    return transform_ROI_coordinates(box, roi)


@pytest.mark.parametrize('result', [[0.074, -0.305, 0.8, 1.5]])
def test_transform_ROI_coordinates(test_results_transform_ROI_coordinates, result):
    values = test_results_transform_ROI_coordinates.numpy()
    np.testing.assert_almost_equal(values[0], result, decimal=4)


@pytest.fixture
def test_results_compute_max_ROI_level():
    area = 0.5
    return compute_max_ROI_level(area)


def test_compute_max_ROI_level(test_results_compute_max_ROI_level):
    values = test_results_compute_max_ROI_level
    assert values == 3


@pytest.fixture
def test_results_compute_scaled_area():
    H, W = tf.Variable([100.]), tf.Variable([100.])
    image_shape = tf.Variable([512, 512])
    return compute_scaled_area(H, W, image_shape)


def test_compute_compute_scaled_area(test_results_compute_scaled_area):
    values = test_results_compute_scaled_area.numpy()
    assert values == 228.57143
