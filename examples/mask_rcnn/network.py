import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Input, Lambda
from tensorflow.keras.models import Model
from mask_rcnn.utils import generate_pyramid_anchors, norm_boxes_graph
from mask_rcnn.utils import fpn_classifier_graph, compute_backbone_shapes
from mask_rcnn.utils import build_fpn_mask_graph
from mask_rcnn.layers import DetectionTargetLayer, ProposalLayer


def get_anchors(config):
    """Returns anchor pyramid for the given image size
    """
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = generate_pyramid_anchors(
        config.RPN_ANCHOR_SCALES,
        config.RPN_ANCHOR_RATIOS,
        backbone_shapes,
        config.BACKBONE_STRIDES,
        config.RPN_ANCHOR_STRIDE)
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
    anchors = Layer(name='anchors')(anchors)
    return anchors


def create_network_head(backbone_model, config):
    image = backbone_model.keras_model.input
    feature_maps = backbone_model.keras_model.output
    rpn_class_logits, rpn_class, rpn_bbox = backbone_model.RPN(feature_maps)
    anchors = get_anchors(config)
    rpn_rois = ProposalLayer(
        proposal_count=config.POST_NMS_ROIS_INFERENCE,
        nms_threshold=config.RPN_NMS_THRESHOLD,rpn_bbox_std_dev= config.RPN_BBOX_STD_DEV,
            pre_nms_limit= config.PRE_NMS_LIMIT, images_per_gpu =config.IMAGES_PER_GPU,
        batch_size=config.BATCH_SIZE,name='ROI')([rpn_class, rpn_bbox, anchors])

    # Groundtruth for detections
    input_rpn_match = Input(shape=[None, 1], name='input_rpn_match',
                            dtype=tf.int32)
    input_rpn_bbox = Input(shape=[None, 4], name='input_rpn_bbox',
                           dtype=tf.float32)
    active_class_ids = tf.zeros([config.NUM_CLASSES], dtype=tf.int32)
    input_gt_class_ids = Input(shape=[None],
                               name='input_gt_class_ids', dtype=tf.int32)
    input_gt_boxes = Input(shape=[None, 4], name='input_gt_boxes',
                           dtype=tf.float32)
    groundtruth_boxes = Lambda(lambda x: norm_boxes_graph(
                               x, K.shape(image)[1:3]))(input_gt_boxes)
    groundtruth_masks = Input(
        shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
        name='input_gt_masks', dtype=bool)

    # Detection targets
    rois, target_class_ids, target_boxes, target_masks = DetectionTargetLayer(
        images_per_gpu=config.IMAGES_PER_GPU, mask_shape=config.MASK_SHAPE,
        train_rois_per_image=config.TRAIN_ROIS_PER_IMAGE,
        roi_positive_ratio=config.ROI_POSITIVE_RATIO,
        bbox_std_dev=config.BBOX_STD_DEV, use_mini_mask=config.USE_MINI_MASK, batch_size=config.BATCH_SIZE,
        name='proposal_targets')([rpn_rois, input_gt_class_ids,groundtruth_boxes,groundtruth_masks])

    # Network heads
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
        rois, feature_maps[:-1], config, train_bn=config.TRAIN_BN,
        fc_layers_size=1024)

    mrcnn_mask = build_fpn_mask_graph(rois, feature_maps[:-1], config,
                                      train_bn=config.TRAIN_BN)
    output_rois = Lambda(lambda x: x * 1, name='output_rois')(rois)

    # Losses
    RPN_output = [rpn_class_logits, rpn_bbox]
    target = [target_class_ids, target_boxes, target_masks]
    predictions = [mrcnn_class_logits, mrcnn_bbox, mrcnn_mask]
    loss_inputs = [RPN_output, target, predictions, active_class_ids]

    inputs = [image, input_rpn_match, input_rpn_bbox, input_gt_class_ids,
              input_gt_boxes, groundtruth_masks]
    outputs = [rpn_class_logits, rpn_class, rpn_bbox,
               mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
               rpn_rois, output_rois]
    model = Model(inputs, outputs,TRAIN_BN=config.TRAIN_BN, IMAGE_SHAPE= config.IMAGE_SHAPE,
                  BACKBONE= config.BACKBONE, TOP_DOWN_PYRAMID_SIZE= config.TOP_DOWN_PYRAMID_SIZE, name='mask_rcnn')
    return model
