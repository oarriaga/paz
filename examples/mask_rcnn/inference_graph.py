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
        self.POST_NMS_ROIS_INFERENCE = config.POST_NMS_ROIS_INFERENCE
        self.RPN_NMS_THRESHOLD = config.RPN_NMS_THRESHOLD
        self.TRAIN_BN = config.TRAIN_BN
        self.FPN_CLASSIF_FC_LAYERS_SIZE = config.FPN_CLASSIF_FC_LAYERS_SIZE

    def __call__(self):
        keras_model = self.model.keras_model
        input_image = keras_model.input
        anchors = Input(shape=[None, 4], name='input_anchors')
        feature_maps = keras_model.output

        rpn_class_logits, rpn_class, rpn_bbox = self.model.RPN(feature_maps)

        rpn_rois = ProposalLayer(
                    proposal_count=self.POST_NMS_ROIS_INFERENCE,
                    nms_threshold=self.RPN_NMS_THRESHOLD,
                    name='ROI',
                    config=self.config)([rpn_class, rpn_bbox, anchors])

        _, classes, mrcnn_bbox = fpn_classifier_graph(rpn_rois,
                                                      feature_maps[:-1],
                                                      mode='inference',
                                                      config=self.config,
                                                      train_bn=self.TRAIN_BN)
        detections = DetectionLayer(self.config, name='mrcnn_detection')(
                            [rpn_rois, classes, mrcnn_bbox])
        detection_boxes = Lambda(lambda x: x[..., :4])(detections)
        mrcnn_mask = build_fpn_mask_graph(detection_boxes, feature_maps[:-1],
                                          self.config, train_bn=self.TRAIN_BN)

        inference_model = Model([input_image, anchors],
                                [detections, classes, mrcnn_bbox,
                                mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                                name='mask_rcnn')
        return inference_model
