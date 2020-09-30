import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from mask_rcnn.utils import fpn_classifier_graph, build_fpn_mask_graph
from mask_rcnn.layers import ProposalLayer, DetectionLayer


class InferenceGraph():
    """Build Inference graph for Mask RCNN

    # Arguments:
        model: Base Mask RCNN model
        config: Instance of base configuration class

    # Returns:
        Inference model
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def __call__(self):
        keras_model = self.model.keras_model
        input_image = keras_model.input
        anchors = Input(shape=[None, 4], name='input_anchors')
        rpn_feature_maps = keras_model.output
        mrcnn_feature_maps = keras_model.output[:-1]

        rpn_class_logits, rpn_class, rpn_bbox = self.model.RPN(rpn_feature_maps)

        rpn_rois = ProposalLayer(
                    proposal_count=self.config.POST_NMS_ROIS_INFERENCE,
                    nms_threshold=self.config.RPN_NMS_THRESHOLD,
                    name='ROI',
                    config=self.config)([rpn_class, rpn_bbox, anchors])

        _, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(rpn_rois, mrcnn_feature_maps,
                                        mode='inference', config=self.config,
                                        train_bn=self.config.TRAIN_BN,
                                        fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)

        detections = DetectionLayer(self.config, name='mrcnn_detection')(
                            [rpn_rois, mrcnn_class, mrcnn_bbox])

        detection_boxes = Lambda(lambda x: x[..., :4])(detections)
        mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                                self.config, train_bn=self.config.TRAIN_BN)

        inference_model = Model([input_image, anchors],
                            [detections, mrcnn_class, mrcnn_bbox,
                             mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')
        return inference_model
