import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda

from ..config import Config
import numpy as np

from ..layers import ProposalLayer, DetectionTargetLayer
from ..layers import DetectionLayer, PyramidROIAlign
from ..utils import norm_boxes_graph, fpn_classifier_graph
from ..model import MaskRCNN
from ..layer_utils import slice_batch, compute_ROI_level, apply_ROI_pooling
from ..layer_utils import rearrange_pooled_features, trim_by_score, \
                          apply_box_delta, clip_image_boundaries
from ..layer_utils import refine_instances, compute_overlaps_graph, \
                          compute_ROI_overlaps, pad_ROI, pad_ROI_priors
from ..layer_utils import update_priors, compute_target_masks, \
                          apply_box_deltas, clip_boxes, NMS
from ..layer_utils import filter_low_confidence, apply_NMS, get_top_detections, zero_pad_detections


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
    return [class_ids, ground_truth_boxes, masks]


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
                         images_per_gpu=config.IMAGES_PER_GPU)\
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
    boxes = apply_box_delta(pre_nms_anchors, deltas, config.IMAGES_PER_GPU)
    boxes = clip_image_boundaries(boxes, config.IMAGES_PER_GPU)
    return boxes


@pytest.fixture
def proposal_layer_NMS(proposal_layer_apply_box_delta, proposal_layer_trim_by_score, config):
    boxes = proposal_layer_apply_box_delta
    scores, __, __ = proposal_layer_trim_by_score

    proposals = slice_batch([boxes, scores], NMS(2000,0.7),
                            config.IMAGES_PER_GPU)
    return proposals


@pytest.fixture
def detection_layer_box_delta_graph(proposal_layer, config, FPN_classifier):
    probs, deltas = FPN_classifier

    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    deltas_specific = tf.gather_nd(deltas, indices)

    refined_rois = apply_box_deltas(
        proposal_layer, deltas_specific * config.BBOX_STD_DEV)
    refined_rois = clip_boxes(refined_rois, config.WINDOW)

    keep = tf.where(class_ids > 0)[:, 0]
    return refined_rois, keep


@pytest.fixture
def detection_layer_box_apply_NMS(detection_layer_box_delta_graph, FPN_classifier,
                                  config):
    probs, deltas = FPN_classifier
    refined_rois, keep = detection_layer_box_delta_graph

    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)

    if config.DETECTION_MIN_CONFIDENCE:
        keep = filter_low_confidence(class_scores, keep,
                                     config.DETECTION_MIN_CONFIDENCE)

    nms_keep = apply_NMS(class_ids, class_scores, refined_rois, keep,
                         config.DETECTION_MAX_INSTANCES,
                         config.DETECTION_NMS_THRESHOLD)

    return nms_keep, class_scores


@pytest.fixture
def detection_layer_box_get_top_detections(detection_layer_box_delta_graph,
                                           detection_layer_box_apply_NMS):
    nms_keep, class_scores = detection_layer_box_apply_NMS
    _, keep = detection_layer_box_delta_graph
    keep = get_top_detections(class_scores, keep, nms_keep,
                              config.DETECTION_MAX_INSTANCES)
    return keep


@pytest.fixture
def detection_layer_refine_detections_graph(detection_layer_box_get_top_detections, FPN_classifier,
                                            detection_layer_box_delta_graph, config):
    refined_rois, keep = detection_layer_box_delta_graph
    probs, deltas = FPN_classifier
    keep = get_top_detections

    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)

    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.cast(tf.gather(class_ids, keep),
                dtype=tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]], axis=1)
    return zero_pad_detections(detections, config.DETECTION_MAX_INSTANCES)


@pytest.fixture
def detection_layer_fun(proposal_layer, FPN_classifier, detection_layer_refine_detections_graph, config):
    rois = proposal_layer
    mrcnn_class, mrcnn_bbox = FPN_classifier
    detections_batch = slice_batch(
        [rois, mrcnn_class, mrcnn_bbox],
        detection_layer_refine_detections_graph,
        config.IMAGES_PER_GPU)
    return tf.reshape(detections_batch,
                      [config.BATCH_SIZE, config.DETECTION_MAX_INSTANCES, 6])


@pytest.fixture
def detection_target_layer(proposal_layer, ground_truth, config):
    class_ids, boxes, masks = ground_truth
    target_layer = DetectionTargetLayer(images_per_gpu= config.IMAGES_PER_GPU,
                                        mask_shape=config.MASK_SHAPE,
                                        train_rois_per_image=config.TRAIN_ROIS_PER_IMAGE,
                                        roi_positive_ratio=config.ROI_POSITIVE_RATIO,
                                        bbox_std_dev=config.BBOX_STD_DEV,
                                        use_mini_mask=config.USE_MINI_MASK
                                        )
    return target_layer([proposal_layer, class_ids, boxes, masks])


@pytest.fixture
def detection_target_layer_refine_instances(proposal_layer, ground_truth, config):
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposal_layer)[0], 0), [proposal_layer],
                  name='roi_assertion'),
    ]
    with tf.control_dependencies(asserts):
        proposal_layer = tf.identity(proposal_layer)

    refined_priors, crowd_boxes = refine_instances(proposal_layer, ground_truth)
    return refined_priors, crowd_boxes


@pytest.fixture
def detection_target_layer_compute_overlaps_graph(proposal_layer,
                                                  detection_target_layer_refine_instances):
    refined_priors, crowd_boxes = detection_target_layer_refine_instances
    _, refined_boxes, _ = refined_priors

    overlaps = compute_overlaps_graph(proposal_layer, refined_boxes)

    return overlaps


@pytest.fixture
def detection_target_layer_compute_ROI_overlaps(proposal_layer,
                                                detection_target_layer_refine_instances,
                                                detection_target_layer_compute_overlaps_graph,
                                                config):
    refined_priors, crowd_boxes = detection_target_layer_refine_instances
    _, refined_boxes, _ = refined_priors
    overlaps = detection_target_layer_compute_overlaps_graph

    positive_indices, positive_rois, negative_rois = \
         compute_ROI_overlaps(proposal_layer, refined_boxes, crowd_boxes,
                              overlaps, config.TRAIN_ROIS_PER_IMAGE, config.ROI_POSITIVE_RATIO)

    return positive_indices, positive_rois, negative_rois


@pytest.fixture
def detection_target_layer_update_priors(detection_target_layer_refine_instances,
                                         detection_target_layer_compute_overlaps_graph,
                                         detection_target_layer_compute_ROI_overlaps,
                                         config):
    refined_priors, crowd_boxes = detection_target_layer_refine_instances
    overlaps = detection_target_layer_compute_overlaps_graph
    positive_indices, positive_rois, negative_rois = detection_target_layer_compute_ROI_overlaps

    deltas, roi_priors = update_priors(overlaps, positive_indices,
                                       positive_rois, refined_priors, config.BBOX_STD_DEV)

    return deltas, roi_priors


@pytest.fixture
def detection_target_compute_target_masks(detection_target_layer_compute_ROI_overlaps,
                                          detection_target_layer_update_priors, config):
    positive_indices, positive_rois, negative_rois = detection_target_layer_compute_ROI_overlaps
    deltas, roi_priors = detection_target_layer_update_priors

    masks = compute_target_masks(positive_rois, roi_priors, config.MASK_SHAPE, config.USE_MINI_MASK)

    return masks


@pytest.fixture
def detection_target_pad_ROI(detection_target_layer_compute_ROI_overlaps,
                             detection_target_layer_update_priors,
                             detection_target_compute_target_masks, config):
    _, positive_rois, negative_rois = detection_target_layer_compute_ROI_overlaps
    deltas, roi_priors = detection_target_layer_update_priors
    masks = detection_target_compute_target_masks

    rois, num_negatives, num_positives = pad_ROI(positive_rois,
                                                 negative_rois, config.TRAIN_ROIS_PER_IMAGE)
    roi_class_ids, deltas, masks = pad_ROI_priors(num_positives, num_negatives, roi_priors,
                                                  deltas, masks)
    return rois, roi_class_ids, deltas, masks


@pytest.fixture
def pyramid_ROI_level(proposal_layer, feature_maps):
    shape= (1024, 1024, 3)
    roi_level= compute_ROI_level(proposal_layer, shape)

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


@pytest.mark.parametrize('ROI_shape', [(3,)])
def test_proposal_layer(proposal_layer, ROI_shape):
    num_coordinates = 4
    assert proposal_layer.shape[2] == num_coordinates
    assert K.shape(proposal_layer).shape, ROI_shape



def test_proposal_layer_nms(proposal_layer_trim_by_score, proposal_layer_apply_box_delta):
    scores, deltas, pre_nms_anchors = proposal_layer_trim_by_score
    boxes = proposal_layer_apply_box_delta
    #proposals = proposal_layer_NMS
    assert deltas.shape[2] == 4
    assert pre_nms_anchors.shape[2] == 4
    assert boxes.shape[2] == 4
    #assert proposals


@pytest.mark.parametrize('shapes', [[(3,), (2,), (3,), (4,)]])
def test_detection_target_layer(detection_target_layer, shapes):
   ROIs, target_class, target_box, target_mask = detection_target_layer
   mask_shape = (28, 28)
   results_shape = [K.shape(ROIs).shape, K.shape(target_class).shape,
                     K.shape(target_box).shape, K.shape(target_mask).shape]
   assert shapes == results_shape
   assert target_mask.shape[-2:] == mask_shape
   assert ROIs.shape[2] == target_box.shape[2] == 4


# def test_detection_target_layer_functions(detection_target_layer_refine_instances,
#                                           detection_target_layer_compute_overlaps_graph,
#                                           detection_target_layer_compute_ROI_overlaps,
#                                           detection_target_pad_ROI):
#     refined_priors, crowd_boxes = detection_target_layer_refine_instances
#     overlaps = detection_target_layer_compute_overlaps_graph
#     positive_indices, positive_rois, negative_rois = detection_target_layer_compute_ROI_overlaps
#     rois, roi_class_ids, deltas, masks = detection_target_pad_ROI
#
#     assert refined_priors
#     assert overlaps
#     assert positive_indices
#     assert deltas
#     assert masks


@pytest.mark.parametrize('shape', [(1, 100, 6)])
def test_detection_layer(proposal_layer, FPN_classifier, config, shape):
    mrcnn_class, mrcnn_bbox = FPN_classifier
    detections = DetectionLayer(batch_size=config.BATCH_SIZE, window=config.WINDOW,
                                bbox_std_dev=config.BBOX_STD_DEV,
                                images_per_gpu=config.IMAGES_PER_GPU,
                                detection_max_instances=config.DETECTION_MAX_INSTANCES,
                                detection_min_confidence=config.DETECTION_MIN_CONFIDENCE,
                                detection_nms_threshold=config.DETECTION_NMS_THRESHOLD,
                                name='mrcnn_detection')\
         ([proposal_layer, mrcnn_class, mrcnn_bbox])

    assert detections.shape == shape



# def test_detection_layer_functions(detection_layer_box_delta_graph, detection_layer_apply_NMS,
#                                    detection_layer_get_top_detections):
#     refined_rois, keep = detection_layer_box_delta_graph
#     nms_keep, class_scores = detection_layer_apply_NMS
#     keep = detection_layer_get_top_detections
#
#     assert refined_rois
#     assert keep
#     assert nms_keep
#     assert class_scores


# def test_detection_layer_FPN_classifier(detection_layer_box_delta_graph):
#     refined_rois, keep = detection_layer_box_delta_graph
#     assert refined_rois

@pytest.mark.parametrize('shape', [(1024, 1024, 3)])
def test_pyramid_ROI_align(proposal_layer, feature_maps, shape):
    ROI_align = PyramidROIAlign([7, 7])([proposal_layer, shape] + feature_maps)

    assert K.int_shape(ROI_align) == (1, None, 7, 7, 256)


@pytest.mark.parametrize('shape', [(1024, 1024, 3)])
def test_pyramid_ROI_align_functions(pyramid_ROI_level, pyramid_ROI_pooling, pyramid_ROI_pooled_features, shape):
    roi_level= pyramid_ROI_level
    pooled, box_to_level = pyramid_ROI_pooling
    pooled = pyramid_ROI_pooled_features

    assert K.int_shape(roi_level) == (1, None)
    assert K.int_shape(box_to_level) == (None, 2)
    assert K.int_shape(pooled) == (1, None, 7, 7, 256)
