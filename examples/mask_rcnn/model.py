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

from mask_rcnn.utils import log, get_resnet_features, build_rpn_model
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Input, Lambda
from tensorflow.keras.models import Model

from mask_rcnn.utils import generate_pyramid_anchors, norm_boxes_graph
from mask_rcnn.utils import fpn_classifier_graph, compute_backbone_shapes
from mask_rcnn.utils import build_fpn_mask_graph
from mask_rcnn.layers import DetectionTargetLayer, ProposalLayer
from mask_rcnn.layer_utils import slice_batch
from mask_rcnn.loss_end_point import ProposalBBoxLoss, ProposalClassLoss,\
    BBoxLoss,ClassLoss,MaskLoss
tf.compat.v1.disable_eager_execution()


class MaskRCNN:
    """Encapsulates the Mask RCNN model functionality.

    # Arguments:
        config: Instance of basic model configurations
        model_dir: Directory to save training logs and weights
    """
    def __init__(self, config, model_dir, train_bn, image_shape, backbone, top_down_pyramid_size):
        self.config = config
        self.model_dir = model_dir
        self.train_bn = train_bn
        self.image_shape = image_shape
        self.get_backbone_features = backbone
        self.fpn_size = top_down_pyramid_size
        self.keras_model = build_backbone(image_shape, backbone, top_down_pyramid_size, train_bn)

    def RPN(self, rpn_feature_maps):
        return RPN_model(self.config, self.fpn_size, rpn_feature_maps)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        get_trainable(self.keras_model, layer_regex, keras_model=None,
                      indent=0, verbose=1)

    def build_complete_network(self):
        image = self.keras_model.input
        config = self.config
        feature_maps = self.keras_model.output
        rpn_class_logits, rpn_class, rpn_bbox = self.RPN(feature_maps)
        rpn_rois = get_rpn_rois(config, rpn_class, rpn_bbox)

        input_rpn_match, input_rpn_bbox, input_gt_class_ids, \
        input_gt_boxes, groundtruth_boxes, groundtruth_masks = get_ground_truth_values(config, image)

        rois, target_class_ids, target_boxes, target_masks = get_detections_target \
            (config, rpn_rois, input_gt_class_ids, groundtruth_boxes, groundtruth_masks)
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, output_rois = \
            create_head(self.keras_model, config, rois)

        RPN_output, target, predictions, active_class_ids, loss_inputs = get_loss(config, mrcnn_class_logits,
                                                                                  mrcnn_bbox, mrcnn_mask,
                                                                                  target_class_ids,
                                                                                  target_boxes, target_masks,
                                                                                  rpn_class_logits, rpn_bbox)
        rpn_class_loss = ProposalClassLoss(config=config, name='rpn_class_loss')\
            (input_rpn_match, rpn_class_logits)

        rpn_bbox_loss= ProposalBBoxLoss(config=config, rpn_match=input_rpn_match, name='rpn_bbox_loss')\
            (input_rpn_bbox, rpn_bbox)
        mrcnn_class_loss = ClassLoss(config=config, active_class_ids=active_class_ids, name='mrcnn_class_loss')\
             (target_class_ids, mrcnn_class_logits)
        mrcnn_bbox_loss = BBoxLoss(config=config, target_class_ids=target_class_ids, name='mrcnn_bbox_loss')\
             (target_boxes, mrcnn_bbox)
        mrcnn_mask_loss = MaskLoss(config, target_class_ids=target_class_ids, name='mrcnn_mask_loss')\
             (target_masks, mrcnn_mask)

        inputs = [image, input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, groundtruth_masks]

        if not config.USE_RPN_ROIS:
            inputs.append(input_rois)

        outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                   mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                   rpn_rois, output_rois,rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss,
                   mrcnn_bbox_loss, mrcnn_mask_loss]

        self.keras_model = Model(inputs, outputs, name='mask_rcnn')


def convolution_block(inputs, filters, stride, name, padd='valid'):
    """Convolution block containing Conv2D
    # Arguments
        inputs: Keras/tensorflow tensor input.
        filters: Int. Number of filters.
        strides: Stride Dimension

    # Returns
        Keras/tensorflow tensor.
    """
    x = Conv2D(filters, stride, padding=padd, name=name)(inputs)
    return x


def upsample_block(y, x, filters, up_sample_name='None', fpn_name='None', fpn_add_name='None'):
    """Upsample block. This block upsamples ``x``, concatenates a
    ``branch`` tensor and applies two convolution blocks:
    Upsample ->  ConvBlock --> Add.

    # Arguments
        x: Keras/tensorflow tensor.
        y: Keras/tensorflow tensor.
        filters: Int. Number of filters.

    # Returns
        A Keras tensor.
    """
    upsample = UpSampling2D(size=(2, 2), name=up_sample_name)(y)
    conv2d = convolution_block(x, filters, (1, 1), name=fpn_name)
    p = Add(name=fpn_add_name)([upsample, conv2d])
    return p


def build_backbone(image_shape, backbone_features, fpn_size, train_bn):
    """Builds ``BACKBONE`` class for mask-RCNN.

    # Arguments
        image_shape: [H,W].
        backbone_features
        fpn_size
        train_bn

    # Returns
        Model.
    """
    height, width = image_shape[:2]
    raise_exception(height, width)
    input_image = Input(shape=[None,None, image_shape[2]], name='input_image')

    C2, C3, C4, C5= get_backbone_features(input_image, backbone_features, train_bn)
    P2, P3, P4, P5, P6 = build_layers(C2, C3, C4, C5, fpn_size)

    model1 = Model([input_image], [P2, P3, P4, P5, P6], name='mask_rcnn_before')

    return model1

def build_layers(C2, C3, C4, C5, fpn_size):
    """Builds `layers for mask-RCNN backbone.

    # Arguments
        C2, C3, C4, C5: channel sizes
        fpn_size

    # Returns
        Model layers.
    """
    P5 = convolution_block(C5, fpn_size, (1, 1), name='fpn_c5p5')
    P4 = upsample_block(P5, C4, fpn_size, up_sample_name='fpn_p5upsampled', fpn_name='fpn_c4p4',
                        fpn_add_name='fpn_p4add')
    P3 = upsample_block(P4, C3, fpn_size, up_sample_name='fpn_p4upsampled', fpn_name='fpn_c3p3',
                        fpn_add_name='fpn_p3add')
    P2 = upsample_block(P3, C2, fpn_size, up_sample_name='fpn_p3upsampled', fpn_name='fpn_c2p2',
                        fpn_add_name='fpn_p2add')

    P2 = convolution_block(P2, fpn_size, (3, 3), name='fpn_p2', padd='SAME')
    P3 = convolution_block(P3, fpn_size, (3, 3), name='fpn_p3', padd='SAME')
    P4 = convolution_block(P4, fpn_size, (3, 3), name='fpn_p4', padd='SAME')
    P5 = convolution_block(P5, fpn_size, (3, 3), name='fpn_p5', padd='SAME')
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)
    return(P2, P3, P4, P5, P6)


def get_backbone_features(image, backbone_features, train_BN):
    """Gets backbone features for mask-RCNN backbone.

    # Arguments
        image
        backbone_features
        train_BN

    # Returns
        Model layers.
    """
    if callable(backbone_features):
        _, C2, C3, C4, C5 = backbone_features(image, stage5=True, train_bn=train_BN)
    else:
        _, C2, C3, C4, C5 = get_resnet_features(image, backbone_features, stage5=True, train_bn=train_BN)

    return(C2, C3, C4, C5)


def raise_exception(height, width):
    """Raise exception when image is not a multiple of 2

    # Arguments
        height, width : size of image

    """
    if height / 2 ** 6 != int(height / 2 ** 6) or width / 2 ** 6 != int(width / 2 ** 6):
        raise Exception('Image size must be dividable by 2 atleast'
                        '6 times')


def get_imagenet_weights():
    """Define path for weights default

    # Returns
        weights_file.
    """
    weight_path = 'https://github.com/fchollet/deep-learning-models/'\
                  'releases/download/v0.2/'\
                  'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    filepath = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_file = get_file(filepath, weight_path, cache_subdir='models',
                            md5_hash='a268eb855778b3df3c7506639542a6af')
    return weights_file


def RPN_model(config, fpn_size, rpn_feature_maps):
    """Build complete region specific network

    # Arguments
        config
        fpn_size
        rpn_feature_masks

    # Returns
        Model output.
    """
    rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                        len(config.RPN_ANCHOR_RATIOS),
                        fpn_size)
    layer_outputs = [rpn([feature]) for feature in rpn_feature_maps]
    names = ['rpn_class_logits', 'rpn_class', 'rpn_bbox']
    outputs = list(zip(*layer_outputs))
    outputs = [Concatenate(axis=1, name=name)(list(output))
               for output, name in zip(outputs, names)]
    return outputs


def get_trainable(model, layer_regex, keras_model=None, indent=0, verbose=1):
    """Build all the layers by selecting the model

    # Arguments
        model
        layer_regex
        keras_model
    # Returns
        Model output.
    """
    if verbose > 0 and keras_model is None:
        log('Selecting layers to train')

    keras_model = keras_model or model
    if hasattr(keras_model, 'inner_model'):
        layers = keras_model.inner_model.layers

    layers = keras_model.layers
    for layer in layers:
        if layer.__class__.__name__ == 'Model':
            get_trainable(layer_regex, keras_model=layer, indent=indent + 4 )
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
    #anchors = Layer(name='anchors')(anchors)
    class ConstLayer(tf.keras.layers.Layer):
        def __init__(self, x, name=None, dtype=np.float32):
            super(ConstLayer, self).__init__(name=name, dtype=dtype)
            self.x = tf.Variable(x)

        def call(self, input):
            return self.x

    anchors = ConstLayer(anchors, name="anchors")(input_image)
    return anchors


def gnd_truth_call(image):
    """Decorator function used to call the norm_boxes_graph function
    """
    shape = tf.shape(image)[1:3]

    def _gnd_truth_call(boxes):
        return norm_boxes_graph(boxes, shape)
    return _gnd_truth_call


def get_ground_truth_values(config, image):
    """Returns the region specific values of groundtruth values needed for network head creation
    """
    rpn_match = Input(shape=[None, 1], name='input_rpn_match', dtype=tf.int32)
    rpn_bbox = Input(shape=[None, 4], name='input_rpn_bbox', dtype=tf.float32)

    class_ids = Input(shape=[None],name='input_gt_class_ids', dtype=tf.int32)
    input_boxes = Input(shape=[None, 4], name='input_gt_boxes', dtype=tf.float32)

    boxes = gnd_truth_call(image)(input_boxes)

    if config.USE_MINI_MASK:
        input_gt_masks = Input(
            shape=[config.MINI_MASK_SHAPE[0],
                   config.MINI_MASK_SHAPE[1], None],
            name="input_gt_masks", dtype=bool)
    else:
        input_gt_masks = Input(
            shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
            name="input_gt_masks", dtype=bool)

    return rpn_match, rpn_bbox, class_ids, \
        input_boxes, boxes, input_gt_masks


def get_rpn_rois(config, rpn_class, rpn_bbox):
    """Returns the output of Proposal layer i.e the ROIs
    """
    anchors = get_anchors(config)
    return ProposalLayer(
        proposal_count=config.POST_NMS_ROIS_TRAINING,
        nms_threshold=config.RPN_NMS_THRESHOLD, rpn_bbox_std_dev=config.RPN_BBOX_STD_DEV,
        pre_nms_limit=config.PRE_NMS_LIMIT, images_per_gpu=config.IMAGES_PER_GPU,
        batch_size=config.BATCH_SIZE, name='ROI')([rpn_class, rpn_bbox, anchors])


def get_detections_target(config, rpn_rois, input_gt_class_ids, groundtruth_boxes, groundtruth_masks):
    """Returns the output of Detection target layer i.e the detections
    from the given proposals
    """
    return DetectionTargetLayer(
        images_per_gpu=config.IMAGES_PER_GPU, mask_shape=config.MASK_SHAPE,
        train_rois_per_image=config.TRAIN_ROIS_PER_IMAGE,
        roi_positive_ratio=config.ROI_POSITIVE_RATIO,
        bbox_std_dev=config.BBOX_STD_DEV, use_mini_mask=config.USE_MINI_MASK, batch_size=config.BATCH_SIZE,
        name='proposal_targets')([rpn_rois, input_gt_class_ids, groundtruth_boxes, groundtruth_masks])


def create_head(backbone_model, config, rois):
    """ Creation of region specific network head
    """
    feature_maps = backbone_model.output
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
        rois, feature_maps[:-1], config, train_bn=config.TRAIN_BN,
        fc_layers_size=1024)

    mrcnn_mask = build_fpn_mask_graph(rois, feature_maps[:-1], config,
                                      train_bn=config.TRAIN_BN)
    output_rois =  call_rois()(rois)
    return mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, output_rois


def get_loss(config, mrcnn_class_logits, mrcnn_bbox, mrcnn_mask, target_class_ids,
             target_boxes, target_masks, rpn_class_logits, rpn_bbox):
    """Returns the total input loss of the network
    """
    active_class_ids = tf.ones([config.NUM_CLASSES], dtype=tf.int32)

    RPN_output = [rpn_class_logits, rpn_bbox]
    target = [target_class_ids, target_boxes, target_masks]
    predictions = [mrcnn_class_logits, mrcnn_bbox, mrcnn_mask]
    loss_inputs = [RPN_output, target, predictions, active_class_ids]

    return RPN_output, target, predictions, active_class_ids, loss_inputs


def call_rois():

    def _call_rois(value):
        return value * 1
    return _call_rois