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
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Activation, TimeDistributed, Lambda

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer

from mask_rcnn.backend.boxes import normalized_boxes

from mask_rcnn.model.layers.detection_target import DetectionTargetLayer
from mask_rcnn.model.layers.proposal import ProposalLayer
from mask_rcnn.model.layers.detection import DetectionLayer

from mask_rcnn.model.layers.bounding_box_loss import BoundingBoxLoss
from mask_rcnn.model.layers.class_loss import ClassLoss
from mask_rcnn.model.layers.mask_loss import MaskLoss

from mask_rcnn.model.layers.feature_pyramid_network import FPN_classifier_graph
from mask_rcnn.model.layers.feature_pyramid_network import build_FPN_mask_graph
from mask_rcnn.model.rpn_model import RPN_model

from mask_rcnn.backend.boxes import generate_pyramid_anchors
from mask_rcnn.pipelines.data_generator import ComputeBackboneShapes
from tensorflow.keras.layers import BatchNormalization as BatchNorm

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    # Arguments:
        model_dir: Directory to save training logs and weights
    """
    def __init__(self, model_dir, image_shape, backbone, batch_size,
                 images_per_gpu, RPN_anchor_scales, train_ROIs_per_image,
                 num_classes, window=None):
        self.model_dir = model_dir
        self.image_shape = np.array(image_shape)
        self.get_backbone = backbone
        self.batch_size = batch_size
        self.images_per_gpu = images_per_gpu
        self.RPN_anchor_scales = RPN_anchor_scales
        self.train_ROIs_per_image = train_ROIs_per_image
        self.window = window
        self.num_classes = num_classes
        self.keras_model = build_backbone(image_shape, backbone, FPN_size=256,
                                          train_bn=False)

    def set_trainable(self, layer_regex, keras_model=None, indent=0,
                      verbose=1):
        """Set the trainability of selected layers within a Keras model.
           This method allows you to selectively set the trainability of layers
           in a Keras model based on a regular expression pattern.

        # Arguments:
            layer_regex: A string specifying a regular expression pattern to
                         match layer names. Layers whose names match the
                         pattern will have their trainability set.
            keras_model: The Keras model in which the layers' trainability
                         should be set. If not provided, the default model
                         associated with the current instance of `self`
                         will be used.
            indent: An integer indicating the number of spaces for indentation
                    in the logging output.
            verbose: An integer representing the verbosity level of the
                    logging output.
        """
        if verbose > 0 and keras_model is None:
            log('Selecting layers to train')

        keras_model = keras_model or self.keras_model

        if hasattr(keras_model, 'inner_model'):
            layers = keras_model.inner_model.layers
        else:
            layers = keras_model.layers

        for layer in layers:
            if layer.__class__.__name__ == 'Model':
                self.set_trainable(layer_regex=layer_regex, keras_model=layer,
                                   indent=indent + 4)
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

    def build_model(self, train=False):
        input_image = self.keras_model.input
        feature_maps = self.keras_model.output

        RPN_class_logits, RPN_class, RPN_box = \
            RPN_model(RPN_anchor_stride=1, RPN_anchor_ratios=[0.5, 1, 2],
                      FPN_size=256, RPN_feature_maps=feature_maps)

        if train:
            # Gather anchors from RPN network
            anchors = build_anchors(self.image_shape, self.RPN_anchor_scales,
                                    backbone="resnet101")
            anchors = np.broadcast_to(anchors, (self.batch_size,) +
                                      anchors.shape)
            # anchors = tf.Variable(anchors, trainable=False)
            anchors = AnchorsLayer(anchors, name='anchors')(input_image)

            RPN_ROIs = ProposalLayer(proposal_count=2000,
                                     nms_threshold=0.7,
                                     RPN_bounding_box_std_dev=np.array([
                                         0.1, 0.1, 0.2, 0.2]),
                                     pre_nms_limit=6000,
                                     images_per_gpu=self.images_per_gpu,
                                     batch_size=self.batch_size,
                                     name='ROI')([RPN_class, RPN_box, anchors])

            # Initialise inputs for training
            input_groundtruth_class_ids, input_groundtruth_boxes,\
                groundtruth_boxes, groundtruth_masks = get_ground_truth_values(
                  self.image_shape, input_image, mini_mask=False)

            ROIs, target_class_ids, target_boxes, target_masks = \
                DetectionTargetLayer(
                    images_per_gpu=self.images_per_gpu,
                    mask_shape=[28, 28],
                    train_ROIs_per_image=self.train_ROIs_per_image,
                    ROI_positive_ratio=0.33,
                    bounding_box_std_dev=[0.1, 0.1, 0.2, 0.2],
                    use_mini_mask=False,
                    batch_size=self.batch_size,
                    name='proposal_targets')(
                    [RPN_ROIs, input_groundtruth_class_ids,
                     groundtruth_boxes, groundtruth_masks])

            mrcnn_class_logits, mrcnn_class, mrcnn_box, mrcnn_mask = \
                create_head(self.keras_model, ROIs,
                            num_classes=self.num_classes,
                            image_shape=self.image_shape)

            mrcnn_class_loss = ClassLoss(self.num_classes)(
                target_class_ids, mrcnn_class_logits)
            mrcnn_box_loss = BoundingBoxLoss()(
                [target_boxes, target_class_ids], mrcnn_box)
            mrcnn_mask_loss = MaskLoss()(
                [target_masks, target_class_ids], mrcnn_mask)

            inputs = [input_image, input_groundtruth_class_ids,
                      input_groundtruth_boxes, groundtruth_masks]

            outputs = [RPN_class_logits, RPN_box, mrcnn_class_loss,
                       mrcnn_box_loss, mrcnn_mask_loss]

        else:
            anchors = Input(shape=[None, 4], name='input_anchors')

            RPN_ROIs = ProposalLayer(proposal_count=1000,
                                     nms_threshold=0.7,
                                     RPN_bounding_box_std_dev=np.array([
                                         0.1, 0.1, 0.2, 0.2]),
                                     pre_nms_limit=6000,
                                     images_per_gpu=self.images_per_gpu,
                                     batch_size=self.batch_size,
                                     name='ROI')([RPN_class, RPN_box, anchors])

            _, classes, mrcnn_box = FPN_classifier_graph(RPN_ROIs,
                                                         feature_maps[:-1],
                                                         self.num_classes,
                                                         self.image_shape)

            detections = DetectionLayer(batch_size=self.batch_size,
                                        bounding_box_std_dev=np.array([
                                            0.1, 0.1, 0.2, 0.2]),
                                        images_per_gpu=self.images_per_gpu,
                                        detection_max_instances=100,
                                        detection_min_confidence=0.7,
                                        detection_nms_threshold=0.3,
                                        image_shape=self.image_shape,
                                        window=self.window,
                                        name='mrcnn_detection')(
                [RPN_ROIs, classes, mrcnn_box])

            detection_boxes = Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_FPN_mask_graph(detection_boxes,
                                              feature_maps[:-1],
                                              self.num_classes,
                                              self.image_shape)
            inputs = [input_image, anchors]
            outputs = [detections, classes, mrcnn_box, mrcnn_mask, RPN_ROIs,
                       RPN_class, RPN_box]

        self.keras_model = Model(inputs=inputs, outputs=outputs,
                                 name='mask_rcnn')


def build_anchors(image_shape, RPN_anchor_scales, backbone=None):
    """Returns a set of anchor boxes for a given image size.

    # Returns:
        anchors : Normalized to match the shape of
        the input image [N, (y1, x1, y2, x2)]
    """
    backbone_shapes = ComputeBackboneShapes()(backbone, image_shape)
    anchor_cache = {}
    if not tuple(image_shape) in anchor_cache:
        anchors = generate_pyramid_anchors(RPN_anchor_scales,
                                           [0.5, 1, 2], backbone_shapes,
                                           [4, 8, 16, 32, 64], 1)
        anchor_cache[tuple(image_shape)] = normalized_boxes(
            anchors, image_shape[:2])
    return anchor_cache[tuple(image_shape)]


def convolution_block(inputs, filters, stride, name, padd='valid'):
    """Convolution block containing Conv2D.

    # Arguments:
        inputs: Keras/tensorflow tensor input.
        filters: Int. Number of filters.
        strides: Stride Dimension
    # Returns:
        Keras/tensorflow tensor.
    """
    x = Conv2D(filters, stride, padding=padd, name=name)(inputs)
    return x


def upsample_block(y, x, filters, up_sample_name='None', FPN_name='None',
                   FPN_add_name='None'):
    """Upsample block. This block upsamples ``x``, concatenates a
    ``branch`` tensor and applies two convolution blocks:
    Upsample ->  ConvBlock --> Add.

    # Arguments:
        x: Keras/tensorflow tensor.
        y: Keras/tensorflow tensor.
        filters: Int. Number of filters.
    # Returns:
        A Keras tensor.
    """
    upsample = UpSampling2D(size=(2, 2), name=up_sample_name)(y)
    apply_filter = convolution_block(x, filters, (1, 1), name=FPN_name)
    p = Add(name=FPN_add_name)([upsample, apply_filter])
    return p


def build_backbone(image_shape, backbone_features, FPN_size, train_bn):
    """Builds the backbone network for Mask-RCNN. Constructs the backbone
    network for Mask-RCNN, which extracts features from the input image and
    produces feature maps at different scales.

    # Arguments:
        image_shape: A tuple representing the shape of the input image in the
                     format (height, width, channels).
        backbone_features: TODO: Provide a description of the backbone features
                           argument.
        FPN_size: TODO: Provide a description of the FPN size argument.
        train_bn: A boolean indicating whether to train the batch normalization
                  layers in the backbone network.

    # Returns:
        model: A Keras model representing the backbone network.
    """
    height, width = image_shape[:2]
    raise_exception(height, width)
    input_image = Input(shape=[None, None, image_shape[2]], name='input_image')

    C2, C3, C4, C5 = get_backbone_features(input_image, backbone_features,
                                           train_bn)
    P2, P3, P4, P5, P6 = build_layers(C2, C3, C4, C5, FPN_size)

    model = Model([input_image], [P2, P3, P4, P5, P6], name='mask_rcnn_before')
    return model


def build_layers(C2, C3, C4, C5, FPN_size):
    """Builds `layers for Mask R-CNN backbone.

    # Arguments:
        C2, C3, C4, C5: channel sizes
        FPN_size: feature pyramid network size
    # Returns:
        Model layers.
    """
    P5 = convolution_block(C5, FPN_size, (1, 1), name='fpn_c5p5')
    P4 = upsample_block(P5, C4, FPN_size, up_sample_name='fpn_p5upsampled',
                        FPN_name='fpn_c4p4', FPN_add_name='fpn_p4add')
    P3 = upsample_block(P4, C3, FPN_size, up_sample_name='fpn_p4upsampled',
                        FPN_name='fpn_c3p3', FPN_add_name='fpn_p3add')
    P2 = upsample_block(P3, C2, FPN_size, up_sample_name='fpn_p3upsampled',
                        FPN_name='fpn_c2p2', FPN_add_name='fpn_p2add')

    P2 = convolution_block(P2, FPN_size, (3, 3), name='fpn_p2', padd='SAME')
    P3 = convolution_block(P3, FPN_size, (3, 3), name='fpn_p3', padd='SAME')
    P4 = convolution_block(P4, FPN_size, (3, 3), name='fpn_p4', padd='SAME')
    P5 = convolution_block(P5, FPN_size, (3, 3), name='fpn_p5', padd='SAME')
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)
    return P2, P3, P4, P5, P6


def get_backbone_features(image, backbone_features, train_bn):
    """Get backbone features for the Mask-RCNN backbone.

    This callable function retrieves the backbone features for the Mask-RCNN
    backbone network. It either calls a provided callable function
    `backbone_features` or uses the default implementation based on ResNet.

    # Arguments:
        image: The input image for which the backbone features should be
               extracted.
        backbone_features: Either a callable function that accepts an image
                           and returns the backbone features, or a string
                           specifying the backbone architecture
                           (e.g., 'resnet50').
        train_bn: A boolean indicating whether to train the batch normalization
                  layers in the backbone network.
    # Returns:
        Model layers.
    """
    if callable(backbone_features):
        _, C2, C3, C4, C5 = backbone_features(image, stage5=True,
                                              train_bn=train_bn)
    else:
        _, C2, C3, C4, C5 = get_resnet_features(image, backbone_features,
                                                stage5=True,
                                                train_bn=train_bn)
    return C2, C3, C4, C5


def raise_exception(height, width):
    """Raise exception when image is not a multiple of 2.

    # Arguments:
        height, width : size of image

    """
    if height / 2 ** 6 != int(height / 2 ** 6) or width / 2 ** 6 != int(
            width / 2 ** 6):
        raise Exception('Image size must be dividable by 2 atleast'
                        '6 times')


def get_resnet_features(input_image, architecture, stage5=True, train_bn=True):
    """Builds ResNet graph.

    # Arguments:
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = ZeroPadding2D((3, 3))(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = Activation('relu')(x)
    C1 = x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),
                   train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',
                       train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',
                            train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',
                   train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',
                       train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',
                       train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',
                            train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',
                   train_bn=train_bn)
    block_count = {'resnet50': 5, 'resnet101': 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i),
                           train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',
                       train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',
                           train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',
                                train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main
                     path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
               use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
               use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = Add()([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main
                     path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with
    subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base +
               '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=use_bias)\
        (input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def get_imagenet_weights():
    """Define path for weights default.

    # Returns:
        weights_file.
    """
    weight_path = 'https://github.com/fchollet/deep-learning-models/'\
                  'releases/download/v0.2/'\
                  'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    filepath = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_file = get_file(filepath, weight_path, cache_subdir='models',
                            md5_hash='a268eb855778b3df3c7506639542a6af')
    return weights_file


def get_ground_truth_values(image_shape, image, mini_mask=False,
                            mini_mask_shape=(56, 56)):
    """Returns groundtruth values needed for network head
    creation of the type required by the model.

    # Arguments:
        input_image: Input image in original form [H, W, C].
    # Returns:
        input class ids: [No. of instances]
        input bounding boxes in normalised form [N, (y1, x1, y2, x2)]
        input groundtruth masks [N, (Shape of Input image/Mini mask shape)]
    """
    class_ids = Input(shape=[None], name='input_gt_class_ids',
                      dtype=tf.int32)
    input_boxes = Input(shape=[None, 4], name='input_gt_boxes',
                        dtype=tf.float32)

    boxes = gnd_truth_call(image)(input_boxes)

    if mini_mask:
        input_groundtruth_masks = Input(
            shape=[mini_mask_shape[0],
                   mini_mask_shape[1], None],
            name="input_gt_masks", dtype=bool)
    else:
        input_groundtruth_masks = Input(
            shape=[image_shape[0], image_shape[1], None],
            name="input_gt_masks", dtype=bool)
    return class_ids, input_boxes, boxes, input_groundtruth_masks


def create_head(backbone_model, ROIs, num_classes, image_shape,
                train_bn=False):
    """ Creation of region specific network head by calling
    RPN classifier and RPN mask models.

    # Arguments:
        backbone model
        ROIs [No. of ROIs before nms (y1, x1, y2, x2)]
    # Return:
        logits: classifier logits (before softmax)
                [batch, num_rois, NUM_CLASSES]
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        box_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                     Deltas to apply to proposal boxes
        Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    feature_maps = backbone_model.output
    mrcnn_class_logits, mrcnn_class, mrcnn_box = \
        FPN_classifier_graph(ROIs, feature_maps[:-1], num_classes,
                             image_shape,
                             train_bn=train_bn, fc_dim=1024)

    mrcnn_mask = build_FPN_mask_graph(ROIs, feature_maps[:-1], num_classes,
                                      image_shape,
                                      train_bn=train_bn)
    return mrcnn_class_logits, mrcnn_class, mrcnn_box, mrcnn_mask


def gnd_truth_call(image):
    """Decorator function used to call the norm_all_boxes function.

    # Arguments:
        image: Input image in original form [H, W, C].
        boxes: Bounding box in original form [N, (y1, x1, y2, x2)].
    # Returns:
        bounding box: Bounding box in normalised form [N, (y1, x1, y2, x2)].
    """
    shape = tf.shape(image)[1:3]

    def _gnd_truth_call(boxes):
        return norm_all_boxes(boxes, shape)

    return _gnd_truth_call


def norm_all_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.

    # Arguments:
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    # Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    H, W = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([H, W, H, W], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),
                                                           array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class AnchorsLayer(Layer):
    def __init__(self, anchors, name="anchors", **kwargs):
        super().__init__(name=name, **kwargs)
        self.anchors = tf.Variable(anchors, trainable=False)

    def call(self, input_image):
        return self.anchors
