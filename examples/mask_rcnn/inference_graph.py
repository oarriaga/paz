import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

from mask_rcnn import utils
from mask_rcnn import layers


class InferenceGraph():
    """Build Inference graph for Mask RCNN
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def __call__(self):
        keras_model = self.model.keras_model
        #input
        input_image = keras_model.input
        # input_image_meta = keras_model.input[1]
        
        #anchors
        anchors = KL.Input(shape=[None, 4], name="input_anchors")

        #Feature maps
        rpn_feature_maps = keras_model.output
        mrcnn_feature_maps = keras_model.output[:-1]

        #RPN Model
        rpn_class_logits, rpn_class, rpn_bbox = self.model.RPN(rpn_feature_maps)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded
        rpn_rois = layers.ProposalLayer(
            proposal_count=self.config.POST_NMS_ROIS_INFERENCE,
            nms_threshold=self.config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=self.config)([rpn_class, rpn_bbox, anchors])

        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                utils.fpn_classifier_graph(rpn_rois, mrcnn_feature_maps,
                                    mode='inference', config=self.config,
                                    train_bn=self.config.TRAIN_BN,
                                    fc_layers_size=self.config.FPN_CLASSIF_FC_LAYERS_SIZE)



        # Detections
        # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
        # normalized coordinates
        
        detections = layers.DetectionLayer(self.config, name="mrcnn_detection")(
            [rpn_rois, mrcnn_class, mrcnn_bbox])
        
        # Create masks for detections
        detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
        mrcnn_mask = utils.build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                                self.config, train_bn=self.config.TRAIN_BN)

        inference_model = KM.Model([input_image, anchors],
                            [detections, mrcnn_class, mrcnn_bbox,
                                mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                            name='mask_rcnn')
        return inference_model
