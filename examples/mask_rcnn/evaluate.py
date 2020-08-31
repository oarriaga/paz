import os
import random
import datetime
import re
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

from mask_rcnn import utils
from mask_rcnn.model import MaskRCNN
from mask_rcnn.layers import AnchorsLayer, DetectionTargetLayer
from mask_rcnn.inference_graph import build_rpn_model, resnet_graph
from mask_rcnn.inference_graph import ProposalLayer, DetectionLayer
from mask_rcnn.inference_graph import build_fpn_mask_graph, fpn_classifier_graph

from paz.models.detection.utils import create_prior_boxes

# Requires TensorFlow 2.0+
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("2.0")
tf.compat.v1.disable_eager_execution()


def inference(model_dir):
    model = MaskRCNN(model_dir)
    _, rpn_class, rpn_bbox = model.outputs
    anchors = KL.Input(shape=[None, 4], name="input_anchors")
    rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

    mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
        fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                config.POOL_SIZE, config.NUM_CLASSES,
                                mode=mode, train_bn=config.TRAIN_BN,
                                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

    # Detections
    # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
    # normalized coordinates
    detections = DetectionLayer(config, name="mrcnn_detection")(
        [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
    
    # Create masks for detections
    detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
    mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                        input_image_meta,
                                        config.MASK_POOL_SIZE,
                                        config.NUM_CLASSES,
                                        train_bn=config.TRAIN_BN)

    model = KM.Model([input_image, input_image_meta, input_anchors],
                        [detections, mrcnn_class, mrcnn_bbox,
                         mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                         name='mask_rcnn')