import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda

from ..config import Config
import numpy as np

from ..layers import ProposalLayer
from ..utils import norm_boxes_graph, fpn_classifier_graph
from ..model import MaskRCNN
from ..layer_utils import slice_batch, compute_ROI_level, apply_ROI_pooling
from ..layer_utils import rearrange_pooled_features, trim_by_score, \
                          apply_box_delta, clip_image_boundaries,compute_delta_specific
from ..layer_utils import detection_targets, compute_NMS, refine_detections,\
                          compute_refined_boxes, compute_keep, compute_refined_rois


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
    boxes = apply_box_delta(pre_nms_anchors, deltas, config.IMAGES_PER_GPU)
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
    class_ids, class_scores, deltas_specific = slice_batch([mrcnn_class, mrcnn_bbox], [],
                                                           compute_delta_specific, config.IMAGES_PER_GPU)
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
def detection_target_layer_compute_refined_boxes(proposal_layer, ground_truth, config):
    class_ids, boxes, masks = ground_truth
    rois = proposal_layer
    refined_boxes, refined_priors, crowd_boxes = compute_refined_boxes(rois,
                                                                       class_ids, boxes, masks)

    return refined_boxes,refined_priors, crowd_boxes


@pytest.fixture
def detection_target_layer_detections_target(proposal_layer,
                                             detection_target_layer_compute_refined_boxes,
                                             config, ground_truth):
    class_ids, boxes, masks = ground_truth
    rois = proposal_layer
    rois, roi_class_ids, deltas, masks= slice_batch([rois, class_ids, boxes, masks],
                                                    [config.TRAIN_ROIS_PER_IMAGE,config.ROI_POSITIVE_RATIO,
                                                     config.MASK_SHAPE,config.USE_MINI_MASK,
                                                     config.BBOX_STD_DEV],
                                                    detection_targets,config.IMAGES_PER_GPU)
    return rois, roi_class_ids, deltas, masks


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


#############################################################################################
###Test functions

###detection_layer
def test_detection_layer_batch(detection_layer_batch):
    detections = detection_layer_batch
    assert detections.shape[0] == 1
    assert detections.shape[2] == 6


def test_detection_layer_fun(detection_layer_compute_keep):
    refined_rois, keep = detection_layer_compute_keep
    assert refined_rois.shape == (1,1000,4)
    assert keep.shape[0] == 1


####Proposal_Layer
def test_proposal_layer_functions(proposal_layer_trim_by_score,
                                  proposal_layer_apply_box_delta):
    scores, deltas, pre_nms_anchors = proposal_layer_trim_by_score
    boxes = proposal_layer_apply_box_delta
    proposals = proposal_layer_NMS
    assert deltas.shape[2] == 4
    assert pre_nms_anchors.shape[2] == 4
    assert boxes.shape[2] == 4
    assert proposals


###detection_target_layer
@pytest.mark.parametrize('shapes', [[(3,), (2,), (3,), (4,)]])
def test_detection_target_batch(detection_target_batch,shapes):
    ROIs, target_class, target_box, target_mask = detection_target_batch
    results_shape = [K.shape(ROIs).shape, K.shape(target_class).shape,
                     K.shape(target_box).shape, K.shape(target_mask).shape]
    assert results_shape == shapes


def test_detection_target_layer_functions(detection_target_layer_compute_refined_boxes,
                                          detection_target_layer_detections_target):
    refined_boxes, refined_priors, crowd_boxes = detection_target_layer_compute_refined_boxes
    rois, roi_class_ids, deltas, masks = detection_target_layer_detections_target
    assert refined_boxes.shape[1] == 4
    assert refined_priors[1].shape[1] == 4
    assert refined_priors[2].shape[1] == 1024
    assert rois.shape[2] == 4
    assert roi_class_ids.shape[0] == 1
    assert deltas.shape[2] == 4
    assert (masks.shape[2], masks.shape[3]) == (28, 28)


####Pyramid_ROI
@pytest.mark.parametrize('shape', [(1024, 1024, 3)])
def test_pyramid_ROI_align_functions(pyramid_ROI_level, pyramid_ROI_pooling,
                                     pyramid_ROI_pooled_features, shape):
    roi_level= pyramid_ROI_level
    pooled, box_to_level = pyramid_ROI_pooling
    pooled = pyramid_ROI_pooled_features
    assert K.int_shape(roi_level) == (1, None)
    assert K.int_shape(box_to_level) == (None, 2)
    assert K.int_shape(pooled) == (1, None, 7, 7, 256)
