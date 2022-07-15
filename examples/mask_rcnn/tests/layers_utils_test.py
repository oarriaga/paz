import pytest
import tensorflow as tf
#tf.enable_eager_execution()
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda

from mask_rcnn.config import Config
import numpy as np

from mask_rcnn.layers import ProposalLayer
from mask_rcnn.utils import norm_boxes_graph, fpn_classifier_graph
from mask_rcnn.model import MaskRCNN
from mask_rcnn.layer_utils import slice_batch, compute_ROI_level, apply_ROI_pooling, pad_ROI,pad_ROI_priors
from mask_rcnn.layer_utils import rearrange_pooled_features, trim_by_score
from mask_rcnn.layer_utils import apply_box_delta, clip_image_boundaries,compute_delta_specific
from mask_rcnn.layer_utils import detection_targets, compute_NMS, refine_detections
from mask_rcnn.layer_utils import compute_refined_boxes, compute_keep, compute_refined_rois
from mask_rcnn.layer_utils import filter_low_confidence, apply_NMS, get_top_detections, compute_IOU
from mask_rcnn.layer_utils import compute_ROI_overlaps,update_priors, compute_target_masks
from mask_rcnn.layer_utils import refine_bbox, trim_zeros, apply_box_deltas, clip_boxes, compute_scaled_area
from mask_rcnn.layer_utils import transform_ROI_coordinates, compute_max_ROI_level

from tensorflow.python.framework.ops import enable_eager_execution, disable_eager_execution

disable_eager_execution()

class MaskRCNNConfig(Config):
    NAME = 'ycb'
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 80 + 1


@pytest.fixture
def config():
    config = MaskRCNNConfig()
    config.WINDOW = norm_boxes_graph((171, 0, 853, 1024), (640, 640))
    return config


@pytest.fixture
def model():
    train_bn = MaskRCNNConfig().TRAIN_BN
    image_shape = MaskRCNNConfig().IMAGE_SHAPE
    backbone = MaskRCNNConfig().BACKBONE
    top_down_pyramid_size = MaskRCNNConfig().TOP_DOWN_PYRAMID_SIZE
    base_model = MaskRCNN(config=MaskRCNNConfig(),
                          model_dir='../../mask_rcnn',train_bn=train_bn,
                          image_shape=image_shape,
                          backbone=backbone,
                          top_down_pyramid_size=top_down_pyramid_size)
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
def FPN_classifier(proposal_layer, feature_maps, config):
    _, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(proposal_layer,
                                                      feature_maps[:-1],
                                                      config=config,
                                                      train_bn=config.TRAIN_BN)
    return mrcnn_class, mrcnn_bbox


@pytest.fixture
def proposal_layer(RPN_model, anchors, config):
    _, RPN_class, RPN_box = RPN_model
    return ProposalLayer(proposal_count=2000, nms_threshold=0.7, name='ROI',
                         rpn_bbox_std_dev=config.RPN_BBOX_STD_DEV,
                         pre_nms_limit=config.PRE_NMS_LIMIT,
                         images_per_gpu=config.IMAGES_PER_GPU,
                         batch_size= config.BATCH_SIZE)\
        ([RPN_class, RPN_box, anchors])


@pytest.fixture
def proposal_layer_trim_by_score(RPN_model, anchors, config):
    _, RPN_class, RPN_box = RPN_model
    RPN_class = RPN_class[:, :, 1]
    RPN_box = RPN_box * np.reshape(config.RPN_BBOX_STD_DEV, [1, 1, 4])
    scores, deltas, pre_nms_anchors = trim_by_score(RPN_class, RPN_box,
                                                    anchors, config.IMAGES_PER_GPU,
                                                    config.PRE_NMS_LIMIT)
    return scores, deltas, pre_nms_anchors


@pytest.fixture
def proposal_layer_apply_box_delta(proposal_layer_trim_by_score,config):
    scores, deltas, pre_nms_anchors = proposal_layer_trim_by_score
    boxes = apply_box_deltas(pre_nms_anchors, deltas, config.IMAGES_PER_GPU)
    boxes = clip_image_boundaries(boxes, config.IMAGES_PER_GPU)
    return boxes


@pytest.fixture
def proposal_layer_NMS(proposal_layer_apply_box_delta, proposal_layer_trim_by_score, config):
    boxes = proposal_layer_apply_box_delta
    scores, __, __ = proposal_layer_trim_by_score
    proposal_count = tf.repeat(2000, config.BATCH_SIZE)
    threshold = tf.repeat(0.7, config.BATCH_SIZE)

    proposals = slice_batch([boxes, scores, proposal_count, threshold], [],compute_NMS,
                            config.IMAGES_PER_GPU)
    return proposals


@pytest.fixture
def detection_layer_batch(proposal_layer, FPN_classifier,config):
    rois = proposal_layer
    mrcnn_class, mrcnn_bbox = FPN_classifier
    detections_batch = slice_batch([rois, mrcnn_class, mrcnn_bbox],
                                   [tf.cast(config.BBOX_STD_DEV, dtype=tf.float32),
                                    config.WINDOW, config.DETECTION_MIN_CONFIDENCE,
                                    config.DETECTION_MAX_INSTANCES,
                                    tf.cast(config.DETECTION_NMS_THRESHOLD, dtype=tf.float32)],
                                   refine_detections, config.IMAGES_PER_GPU)
    return detections_batch


@pytest.fixture
def detection_layer_compute_delta_specific(proposal_layer, FPN_classifier, config):
    mrcnn_class, mrcnn_bbox = FPN_classifier
    class_ids, class_scores, deltas_specific = slice_batch([mrcnn_class, mrcnn_bbox],
                                                           [],compute_delta_specific,
                                                           config.IMAGES_PER_GPU)
    return class_ids, class_scores, deltas_specific


@pytest.fixture
def detection_layer_compute_refined_rois(proposal_layer, detection_layer_compute_delta_specific,
                                         config):
    rois = proposal_layer
    class_ids, class_scores, deltas_specific = detection_layer_compute_delta_specific
    refined_rois = slice_batch([rois, deltas_specific * config.BBOX_STD_DEV], [config.WINDOW],
                               compute_refined_rois,config.IMAGES_PER_GPU)
    return refined_rois


@pytest.fixture
def detection_layer_compute_keep(detection_layer_compute_refined_rois
                                 ,detection_layer_compute_delta_specific, config):

    class_ids, class_scores, deltas_specific = detection_layer_compute_delta_specific
    refine_rois = detection_layer_compute_refined_rois
    keep= slice_batch([class_ids, class_scores,refine_rois],[config.DETECTION_MIN_CONFIDENCE,
                                                             config.DETECTION_MAX_INSTANCES,
                                                             config.DETECTION_NMS_THRESHOLD],
                      compute_keep, config.IMAGES_PER_GPU)
    return refine_rois, keep


@pytest.fixture
def detection_layer_filter_low_confidence(detection_layer_compute_delta_specific, config):
    class_ids, class_scores, deltas_specific = detection_layer_compute_delta_specific

    keep = tf.where(class_ids > 0)[:, 0]
    if config.DETECTION_MIN_CONFIDENCE:
        keep = slice_batch([class_scores],[ keep, config.DETECTION_MIN_CONFIDENCE],
                           filter_low_confidence, config.IMAGES_PER_GPU)
    return keep


@pytest.fixture
def detection_layer_apply_nms(detection_layer_compute_delta_specific,
                                          detection_layer_filter_low_confidence,
                                          detection_layer_compute_refined_rois,
                                          config):
    class_ids, class_scores, deltas_specific = detection_layer_compute_delta_specific
    refined_rois= detection_layer_compute_refined_rois
    keep = detection_layer_filter_low_confidence

    nms_keep = slice_batch([class_ids, class_scores, refined_rois, keep],
                           [config.DETECTION_MAX_INSTANCES,config.DETECTION_NMS_THRESHOLD],
                           apply_NMS, config.IMAGES_PER_GPU)
    return nms_keep


@pytest.fixture
def detection_layer_get_top_detections(detection_layer_compute_delta_specific,
                                       detection_layer_filter_low_confidence,
                                       detection_layer_apply_nms,
                                       config):
    class_ids, class_scores, deltas_specific = detection_layer_compute_delta_specific
    nms_keep = detection_layer_apply_nms
    keep = detection_layer_filter_low_confidence

    keep = slice_batch([class_scores, keep, nms_keep],
                           [config.DETECTION_MAX_INSTANCES],
                           get_top_detections, config.IMAGES_PER_GPU)
    return keep


@pytest.fixture
def detection_target_batch(proposal_layer, config, ground_truth):
    rois = proposal_layer
    class_ids, boxes, masks = ground_truth
    names = ['rois', 'target_class_ids', 'target_bbox', 'target_mask']
    outputs = slice_batch([rois, class_ids, boxes, masks],
                          [config.TRAIN_ROIS_PER_IMAGE,config.ROI_POSITIVE_RATIO,
                           config.MASK_SHAPE, config.MINI_MASK_SHAPE,
                           tf.cast(config.BBOX_STD_DEV, dtype=tf.float32)],
                          detection_targets, config.IMAGES_PER_GPU, names=names)
    return outputs

@pytest.fixture
def detection_target_layer_detections_target(proposal_layer,
                                             config, ground_truth):
    class_ids, boxes, masks = ground_truth
    rois = proposal_layer
    rois, roi_class_ids, deltas, masks= slice_batch([rois, class_ids, boxes, masks],
                                                    [config.TRAIN_ROIS_PER_IMAGE,
                                                     config.ROI_POSITIVE_RATIO,
                                                     config.MASK_SHAPE,config.USE_MINI_MASK,
                                                     config.BBOX_STD_DEV],
                                                    detection_targets,config.IMAGES_PER_GPU)
    return rois, roi_class_ids, deltas, masks


@pytest.fixture
def detection_target_layer_compute_refined_boxes(proposal_layer, ground_truth, config):
    class_ids, boxes, masks = ground_truth
    rois = proposal_layer

    refined_boxes, refined_class_ids, refined_masks, crowd_boxes = \
                                                 slice_batch([rois, class_ids, boxes,masks],
                                                             [],compute_refined_boxes,
                                                              config.IMAGES_PER_GPU)
    return refined_boxes, refined_class_ids, refined_masks, crowd_boxes


@pytest.fixture
def detection_target_layer_compute_IOU(proposal_layer,detection_target_layer_compute_refined_boxes,
                                       config):
    rois = proposal_layer
    refined_boxes, __, __ ,__= detection_target_layer_compute_refined_boxes

    overlaps = slice_batch([rois,refined_boxes], [], compute_IOU, config.IMAGES_PER_GPU)
    return overlaps


@pytest.fixture
def detection_target_layer_compute_ROI_overlap(proposal_layer,
                                               detection_target_layer_compute_refined_boxes,
                                               detection_target_layer_compute_IOU,
                                               config):
    rois = proposal_layer
    refined_boxes, __, __, crowd_boxes = detection_target_layer_compute_refined_boxes
    overlaps  = detection_target_layer_compute_IOU

    positive_indices, positive_rois, negative_rois = slice_batch ([rois, refined_boxes,
                                                                    crowd_boxes, overlaps],
                                                                  [config.TRAIN_ROIS_PER_IMAGE,
                                                                   config.ROI_POSITIVE_RATIO],
                                                                  compute_ROI_overlaps ,
                                                                   config.IMAGES_PER_GPU)
    return positive_indices, positive_rois, negative_rois


@pytest.fixture
def detection_target_layer_update_priors(detection_target_layer_compute_refined_boxes,
                                        detection_target_layer_compute_IOU,
                                        detection_target_layer_compute_ROI_overlap,
                                        config):
    refined_boxes, refined_class_ids, refined_masks,__ = detection_target_layer_compute_refined_boxes
    overlaps  = detection_target_layer_compute_IOU
    positive_indices, positive_rois, negative_rois = detection_target_layer_compute_ROI_overlap

    deltas, roi_prior_class_ids, roi_prior_boxes, roi_masks = \
                                     slice_batch([overlaps, positive_indices, positive_rois,
                                                  refined_class_ids,refined_boxes, refined_masks],
                                                 [config.BBOX_STD_DEV],
                                                 update_priors, config.IMAGES_PER_GPU)

    return deltas, roi_prior_class_ids, roi_prior_boxes, roi_masks


@pytest.fixture
def detection_target_layer_target_masks(detection_target_layer_compute_ROI_overlap,
                                        detection_target_layer_update_priors,
                                        config):

    positive_indices, positive_rois, negative_rois = detection_target_layer_compute_ROI_overlap
    deltas, roi_prior_class_ids, roi_prior_boxes, roi_masks = detection_target_layer_update_priors

    masks = slice_batch([positive_rois, roi_prior_class_ids, roi_prior_boxes, roi_masks],
                        [config.MASK_SHAPE, config.USE_MINI_MASK],
                        compute_target_masks, config.IMAGES_PER_GPU)
    return masks


@pytest.fixture
def detection_target_pad_ROI(detection_target_layer_compute_ROI_overlap,
                             config):
    positive_indices, positive_rois, negative_rois = detection_target_layer_compute_ROI_overlap
    rois, num_negatives, num_positives = slice_batch([positive_rois, negative_rois],
                                                     [config.TRAIN_ROIS_PER_IMAGE],
                                                     pad_ROI, config.IMAGES_PER_GPU)
    return rois, num_negatives, num_positives


@pytest.fixture
def detection_target_pad_ROI_priors(detection_target_layer_update_priors,
                                    detection_target_layer_target_masks,
                                    detection_target_layer_compute_refined_boxes,
                                    detection_target_pad_ROI,
                                    config):

    refined_boxes, refined_class_ids, refined_masks, __ = \
                                    detection_target_layer_compute_refined_boxes
    deltas, __, __, __ = detection_target_layer_update_priors
    masks = detection_target_layer_target_masks
    __,num_negatives, num_positives = detection_target_pad_ROI

    roi_class_ids, deltas, masks = slice_batch([num_positives, num_negatives,
                                                refined_class_ids, deltas, masks],
                                               [],pad_ROI_priors, config.IMAGES_PER_GPU)
    return roi_class_ids, deltas, masks


@pytest.fixture
def pyramid_ROI_level(proposal_layer, feature_maps):
    shape = (1024, 1024, 3)
    roi_level = compute_ROI_level(proposal_layer, shape)
    return roi_level


@pytest.fixture
def pyramid_ROI_pooling(proposal_layer, pyramid_ROI_level, feature_maps):
    roi_level = pyramid_ROI_level
    pooled, box_to_level = apply_ROI_pooling(roi_level, proposal_layer,
                                             feature_maps, [7,7])
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

    assert refined_rois.shape == (1,1000,4)
    assert result.shape[0] == 1
    assert nms_keep.shape[0]== 1
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
def test_detection_target_batch(detection_target_batch,shapes):
    ROIs, target_class, target_box, target_mask = detection_target_batch
    results_shape = [K.shape(ROIs).shape, K.shape(target_class).shape,
                     K.shape(target_box).shape, K.shape(target_mask).shape]
    assert results_shape == shapes


def test_detection_target_layer_functions(detection_target_layer_compute_refined_boxes,
                                          detection_target_layer_detections_target,
                                          detection_target_layer_compute_IOU,
                                          detection_target_layer_compute_ROI_overlap,
                                          detection_target_layer_update_priors,
                                          detection_target_layer_target_masks,
                                          detection_target_pad_ROI,
                                          detection_target_pad_ROI_priors):
    refined_boxes, refined_classes, refined_masks, crowd_boxes = detection_target_layer_compute_refined_boxes
    rois, roi_class_ids, deltas, masks = detection_target_layer_detections_target
    overlaps = detection_target_layer_compute_IOU
    positive_indices, positive_rois, negative_rois = detection_target_layer_compute_ROI_overlap

    deltas, roi_prior_class_ids, roi_prior_boxes, roi_masks = detection_target_layer_update_priors
    masks = detection_target_layer_target_masks
    rois, num_negatives, num_positives = detection_target_pad_ROI
    roi_class_ids, deltas, masks = detection_target_pad_ROI_priors

    assert rois.shape[2] == 4
    assert roi_class_ids.shape[0] == 1
    assert deltas.shape[2] == 4
    assert (masks.shape[2], masks.shape[3]) == (28, 28)
    assert refined_boxes.shape[2] == 4
    assert overlaps.shape[0]  == 1
    assert positive_rois.shape[2] == 4
    assert negative_rois.shape[2] == 4
    assert deltas.shape[2] == 4
    assert roi_prior_boxes.shape[2] == 4
    assert (roi_masks.shape[2],roi_masks.shape[3],roi_masks.shape[4]) == (1024, 1024, 1)
    assert num_negatives.shape[0] == 1
    assert num_positives.shape[0] == 1


@pytest.mark.parametrize('shape', [(1024, 1024, 3)])
def test_pyramid_ROI_align_functions(pyramid_ROI_level, pyramid_ROI_pooling,
                                     pyramid_ROI_pooled_features, shape):
    roi_level= pyramid_ROI_level
    pooled, box_to_level = pyramid_ROI_pooling
    pooled = pyramid_ROI_pooled_features

    assert K.int_shape(roi_level) == (1, None)
    assert K.int_shape(box_to_level) == (None, 2)
    assert K.int_shape(pooled) == (1, None, 7, 7, 256)


@pytest.fixture
def test_results_refine_box():
    box = tf.Variable([[ 337,  661,  585,  969]])
    prior_box = tf.Variable([[350,  650,  590,  1000]])
    return refine_bbox(box,prior_box)


@pytest.mark.parametrize('delta_box', [[ 0.03629032, 0.03246753, -0.03278985, 0.12783337]])
def test_refine_box(test_results_refine_box, delta_box):
    prior_results = test_results_refine_box.numpy()
    np.testing.assert_almost_equal(prior_results[0],delta_box, decimal=6)


@pytest.fixture
def test_results_trim_zeros():
    box = tf.Variable([[0, 0, 0, 0], [0, 0, 0, 0],
                       [337,  661,  585,  969],[337,  661,  585,  969],
                       [337, 661, 585, 969],[337,  661,  585,  969],
                       [337, 661, 585, 969], [337,  661,  585,  969],
                       [0, 0, 0, 0], [0, 0, 0, 0]])
    return trim_zeros(box)

@pytest.mark.parametrize('boxes', [[[337,  661,  585,  969],[337,  661,  585,  969],
                       [337, 661, 585, 969],[337,  661,  585,  969],
                       [337, 661, 585, 969], [337,  661,  585,  969]]])
def test_trim_zeros(test_results_trim_zeros, boxes):
    box, non_zeros= test_results_trim_zeros
    np.testing.assert_almost_equal(box.numpy(), boxes)
    assert non_zeros.numpy().sum() == 6


@pytest.fixture
def test_results_apply_box_delta():
    box = tf.Variable([[337., 661., 500., 300.]])
    deltas = tf.Variable([[ 0.03629032, 0.03246753, -0.03278985, 0.12783337]])
    return apply_box_delta(box, deltas)


@pytest.mark.parametrize('result', [[345.54434, 673.8929,  503.28625, 263.66562]])
def test_apply_box_delta(test_results_apply_box_delta,result):
    values= test_results_apply_box_delta.numpy()
    np.testing.assert_almost_equal(values[0], result, decimal=4)


@pytest.fixture
def test_results_clip_boxes():
    box = tf.Variable([[337., 661., 700., 300.]])
    windows = np.array([400, 500, 800, 900], dtype=np.float32)
    return clip_boxes(box, windows)


@pytest.mark.parametrize('result', [[400., 661., 700., 500.]])
def test_clip_boxes(test_results_clip_boxes,result):
    values= test_results_clip_boxes.numpy()
    np.testing.assert_almost_equal(values[0], result, decimal=4)


@pytest.fixture
def test_results_transform_ROI_coordinates():
    box = tf.Variable([[337., 661., 700., 300.]])
    roi = tf.Variable([[300., 600., 800., 400.]])
    return transform_ROI_coordinates(box, roi)


@pytest.mark.parametrize('result', [[ 0.074, -0.305,  0.8, 1.5]])
def test_transform_ROI_coordinates(test_results_transform_ROI_coordinates, result):
    values= test_results_transform_ROI_coordinates.numpy()
    np.testing.assert_almost_equal(values[0], result, decimal=4)

@pytest.fixture
def test_results_compute_max_ROI_level():
    area = 0.5
    return compute_max_ROI_level(area)


def test_compute_max_ROI_level(test_results_compute_max_ROI_level):
    values= test_results_compute_max_ROI_level
    assert values == 3


@pytest.fixture
def test_results_compute_scaled_area():
    H, W = tf.Variable([100.]),tf.Variable([100.])
    image_shape =tf.Variable([512,512])
    return compute_scaled_area(H,W,image_shape)


def test_compute_compute_scaled_area(test_results_compute_scaled_area):
    values = test_results_compute_scaled_area.numpy()
    assert values == 228.57143