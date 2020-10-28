"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import re
import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Input, Add, Conv2D, Concatenate
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model

from mask_rcnn.utils import log
from mask_rcnn.utils import get_resnet_features, build_rpn_model
tf.compat.v1.disable_eager_execution()


class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    # Arguments:
        config: Instance of basic model configurations
        model_dir: Directory to save training logs and weights
    """

    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.TRAIN_BN = config.TRAIN_BN
        self.IMAGE_SHAPE = config.IMAGE_SHAPE
        self.get_backbone_features = config.BACKBONE
        self.FPN_SIZE = config.TOP_DOWN_PYRAMID_SIZE
        self.keras_model = self.build()

    def build(self):
        H, W = self.IMAGE_SHAPE[:2]
        if H / 2**6 != int(H / 2**6) or W / 2**6 != int(W / 2**6):
            raise Exception('Image size must be dividable by 2 atleast'
                            '6 times')

        input_image = Input(shape=[None, None, self.IMAGE_SHAPE[2]],
                            name='input_image')
        # input_image_meta = Input(shape=[self.config.IMAGE_META_SIZE],
        #                             name="input_image_meta")
        # if mode == "training":
        #     # RPN GT
        #     input_rpn_match = Input(
        #         shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        #     input_rpn_bbox = Input(
        #         shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        #     # Detection GT (class IDs, bounding boxes, and masks)
        #     # 1. GT Class IDs (zero padded)
        #     input_gt_class_ids = Input(
        #         shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        #     # 2. GT Boxes in pixels (zero padded)
        #     # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        #     input_gt_boxes = Input(
        #         shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        #     # Normalize coordinates
        #     gt_boxes = Lambda(lambda x: norm_boxes_graph(
        #         x, K.shape(input_image)[1:3]))(input_gt_boxes)
        #     # 3. GT Masks (zero padded)
        #     # [batch, height, width, MAX_GT_INSTANCES]
        #     if config.USE_MINI_MASK:
        #         input_gt_masks = Input(
        #             shape=[config.MINI_MASK_SHAPE[0],
        #                    config.MINI_MASK_SHAPE[1], None],
        #             name="input_gt_masks", dtype=bool)
        #     else:
        #         input_gt_masks = Input(
        #             shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
        #             name="input_gt_masks", dtype=bool)

        if callable(self.get_backbone_features):
            _, C2, C3, C4, C5 = self.get_backbone_features(
                        input_image, stage5=True, train_bn=self.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = get_resnet_features(input_image,
                                                    self.get_backbone_features,
                                                    stage5=True,
                                                    train_bn=self.TRAIN_BN)

        P5 = Conv2D(self.FPN_SIZE, (1, 1), name='fpn_c5p5')(C5)
        upsample_P5 = UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5)
        conv2d_P4 = Conv2D(self.FPN_SIZE, (1, 1), name='fpn_c4p4')(C4)
        P4 = Add(name='fpn_p4add')([upsample_P5, conv2d_P4])

        upsample_P4 = UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4)
        conv2d_P3 = Conv2D(self.FPN_SIZE, (1, 1), name='fpn_c3p3')(C3)
        P3 = Add(name='fpn_p3add')([upsample_P4, conv2d_P3])

        upsample_P3 = UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3)
        conv2d_P2 = Conv2D(self.FPN_SIZE, (1, 1), name='fpn_c2p2')(C2)
        P2 = Add(name='fpn_p2add')([upsample_P3, conv2d_P2])

        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = Conv2D(self.FPN_SIZE, (3, 3), padding='SAME', name='fpn_p2')(P2)
        P3 = Conv2D(self.FPN_SIZE, (3, 3), padding='SAME', name='fpn_p3')(P3)
        P4 = Conv2D(self.FPN_SIZE, (3, 3), padding='SAME', name='fpn_p4')(P4)
        P5 = Conv2D(self.FPN_SIZE, (3, 3), padding='SAME', name='fpn_p5')(P5)
        P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)

        model = Model([input_image], [P2, P3, P4, P5, P6], name='mask_rcnn')

        # Anchors
        # if mode == "training":
        #     anchors = self.get_anchors(config.IMAGE_SHAPE)
        #     # Duplicate across the batch dimension because Keras requires it
        #     # TODO: can this be optimized to avoid duplicating the anchors?
        #     anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            
        #     #anchor_layer = AnchorsLayer(name='anchors')
        #     anchors = Layer(name='anchors')(anchors)
        #     # A hack to get around Keras's bad support for constants
        #     #anchors = Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        # else:
        #     anchors = input_anchors

        # if mode == "training":
        #     # Class ID mask to mark class IDs supported by the dataset the image
        #     # came from.
        #     active_class_ids = Lambda(
        #         lambda x: parse_image_meta_graph(x)["active_class_ids"]
        #         )(input_image_meta)

        #     if not config.USE_RPN_ROIS:
        #         # Ignore predicted ROIs and use ROIs provided as an input.
        #         input_rois = Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
        #                               name="input_roi", dtype=np.int32)
        #         # Normalize coordinates
        #         target_rois = Lambda(lambda x: norm_boxes_graph(
        #             x, K.shape(input_image)[1:3]))(input_rois)
        #     else:
        #         target_rois = rpn_rois

        #     # Generate detection targets
        #     # Subsamples proposals and generates target outputs for training
        #     # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        #     # padded. Equally, returned rois and targets are zero padded.
        #     rois, target_class_ids, target_bbox, target_mask =\
        #         DetectionTargetLayer(config, name="proposal_targets")([
        #             target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        #     # Network Heads
        #     # TODO: verify that this handles zero padded ROIs
        #     mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
        #         fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
        #                              config.POOL_SIZE, config.NUM_CLASSES,
        #                              mode=mode, train_bn=config.TRAIN_BN,
        #                              fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        #     mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
        #                                       input_image_meta,
        #                                       config.MASK_POOL_SIZE,
        #                                       config.NUM_CLASSES,
        #                                       train_bn=config.TRAIN_BN)

        #     # TODO: clean up (use tf.identify if necessary)
        #     output_rois = Lambda(lambda x: x * 1, name="output_rois")(rois)

        #     # Losses
        #     RPN_output = [rpn_class_logits, rpn_bbox]
        #     target = [target_class_ids, target_bbox, target_mask]
        #     predictions = [mrcnn_class_logits, mrcnn_bbox, mrcnn_mask]
        #     loss = Loss(RPN_output, target, predictions, active_class_ids)
        #     loss = loss.compute_loss()
        #     rpn_class_loss = Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
        #         [input_rpn_match, rpn_class_logits])
        #     rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
        #         [input_rpn_bbox, input_rpn_match, rpn_bbox])
        #     class_loss = Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
        #         [target_class_ids, mrcnn_class_logits, active_class_ids])
        #     bbox_loss = Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
        #         [target_bbox, target_class_ids, mrcnn_bbox])
        #     mask_loss = Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
        #         [target_mask, target_class_ids, mrcnn_mask])

        #     # Model
        #     inputs = [input_image, input_image_meta,
        #               input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
        #     if not config.USE_RPN_ROIS:
        #         inputs.append(input_rois)
        #     outputs = [rpn_class_logits, rpn_class, rpn_bbox,
        #                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
        #                rpn_rois, output_rois,
        #                rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
        #     model = Model(inputs, outputs, name='mask_rcnn')
        return model

    def RPN(self, rpn_feature_maps):
        rpn = build_rpn_model(self.config.RPN_ANCHOR_STRIDE,
                              len(self.config.RPN_ANCHOR_RATIOS),
                              self.FPN_SIZE)
        layer_outputs = [rpn([feature]) for feature in rpn_feature_maps]
        names = ['rpn_class_logits', 'rpn_class', 'rpn_bbox']
        outputs = list(zip(*layer_outputs))
        outputs = [Concatenate(axis=1, name=name)(list(output))
                   for output, name in zip(outputs, names)]
        return outputs

    def get_imagenet_weights(self):
        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/'\
                       'releases/download/v0.2/'\
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        filepath = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_file = get_file(filepath, WEIGHTS_PATH,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_file

    def set_trainable(self, layer_regex, keras_model=None,
                      indent=0, verbose=1):
        """Sets model layers as trainable if their names match
            the given regular expression.

        # Arguments:
            layer_regex: Pre-defined layer regular expressions
                         Select 'heads', '3+', '4+', '5+' or 'all'
            keras_model: Mask RCNN model
        """
        
        if verbose > 0 and keras_model is None:
            log('Selecting layers to train')

        keras_model = keras_model or self.keras_model
        if hasattr(keras_model, 'inner_model'):
            layers = keras_model.inner_model.layers
        layers = keras_model.layers
        for layer in layers:
            if layer.__class__.__name__ == 'Model':
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue
            if not layer.weights:
                continue
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))
