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
from tensorflow.python.keras.layers import Input, Add, Conv2D, Concatenate
from tensorflow.python.keras.layers import UpSampling2D, MaxPooling2D

from mask_rcnn.utils import log, get_resnet_features, build_rpn_model
import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Layer, Input, Lambda
from tensorflow.python.keras.models import Model


from mask_rcnn.utils import generate_pyramid_anchors, norm_boxes_graph
from mask_rcnn.utils import fpn_classifier_graph, compute_backbone_shapes
from mask_rcnn.utils import build_fpn_mask_graph, norm_boxes
from mask_rcnn.layers import DetectionTargetLayer, ProposalLayer, AnchorsLayer, DetectionLayer
from mask_rcnn.loss_end_point import ProposalBBoxLoss, ProposalClassLoss,\
    BBoxLoss, ClassLoss, MaskLoss, L2RegLoss
from tensorflow.python.framework.ops import enable_eager_execution, disable_eager_execution

disable_eager_execution()


class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    # Arguments:
        model_dir: Directory to save training logs and weights
    """
    def __init__(self, model_dir, image_shape, backbone, batch_size, images_per_gpu,
                 rpn_anchor_scales, train_rois_per_image, num_classes, window=None):
        self.model_dir = model_dir
        self.image_shape = image_shape
        self.get_backbone = backbone
        self.batch_size = batch_size
        self.images_per_gpu = images_per_gpu
        self.rpn_anchor_scales = rpn_anchor_scales
        self.train_rois_per_image = train_rois_per_image
        self.window = window
        self.num_classes = num_classes
        self.keras_model = build_backbone(image_shape, backbone, fpn_size=256, train_bn=False)

    def RPN(self, rpn_feature_maps):
        return rpn_model(rpn_anchor_stride=1, rpn_anchor_ratios=[0.5, 1, 2],
                         fpn_size=256, rpn_feature_maps=rpn_feature_maps)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Build all the layers by selecting the model.

            # Arguments:
                model
                layer_regex
                keras_model
            # Returns:
                Model output.
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

    def build_train_model(self):
        image = self.keras_model.input
        RPN_class_logits, RPN_class, RPN_bbox = \
            self.RPN(self.keras_model.output)

        anchors = get_anchors(image_shape, self.rpn_anchor_scales, backbone="resnet101")
        anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)
        anchors = AnchorsLayer(anchors, name='anchors')(image)

        rpn_ROIs = ProposalLayer(proposal_count=2000,
                                 nms_threshold=0.7,
                                 rpn_bbox_std_dev=[0.1, 0.1, 0.2, 0.2],
                                 pre_nms_limit=6000,
                                 images_per_gpu=self.images_per_gpu,
                                 batch_size=self.batch_size, name='ROI')([RPN_class, RPN_bbox, anchors])
        input_gt_class_ids, input_gt_boxes, groundtruth_boxes, \
            groundtruth_masks = get_ground_truth_values(self.image_shape, image, mini_mask=False)

        ROIs, target_class_ids, target_boxes, target_masks = \
            DetectionTargetLayer(images_per_gpu=self.images_per_gpu,
                                 mask_shape=[28, 28],
                                 train_rois_per_image=self.train_rois_per_image,
                                 roi_positive_ratio=0.33,
                                 bbox_std_dev=[0.1, 0.1, 0.2, 0.2],
                                 use_mini_mask=False,
                                 batch_size=self.batch_size, name='proposal_targets')(
                [rpn_ROIs, input_gt_class_ids, groundtruth_boxes, groundtruth_masks])

        mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, output_ROIs = \
            create_head(self.keras_model, ROIs, num_classes=self.num_classes, image_shape=self.image_shape)

        mrcnn_class_loss = ClassLoss(num_classes=num_classes).call(target_class_ids, mrcnn_class_logits)
        mrcnn_bbox_loss = BBoxLoss().call([target_boxes, target_class_ids], mrcnn_bbox)
        mrcnn_mask_loss = MaskLoss().call([target_masks, target_class_ids], mrcnn_mask)

        inputs = [image, input_gt_class_ids, input_gt_boxes, groundtruth_masks]

        outputs = [RPN_class_logits, RPN_bbox, mrcnn_class_loss,
                   mrcnn_bbox_loss, mrcnn_mask_loss]

        self.keras_model = Model(inputs=inputs, outputs=outputs, name='mask_rcnn')

    def build_inference_model(self, config):
        keras_model = self.keras_model
        input_image = keras_model.input
        anchors = Input(shape=[None, 4], name='input_anchors')
        feature_maps = keras_model.output

        rpn_class_logits, rpn_class, rpn_bbox = self.RPN(feature_maps)
        rpn_rois = ProposalLayer(
            proposal_count=1000,
            nms_threshold=0.7, rpn_bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
            pre_nms_limit=6000, images_per_gpu=self.images_per_gpu,
            batch_size=self.batch_size, name='ROI')([rpn_class, rpn_bbox, anchors])

        _, classes, mrcnn_bbox = fpn_classifier_graph(rpn_rois, feature_maps[:-1],
                                                      num_classes=self.num_classes,
                                                      image_shape=self.image_shape)

        detections = DetectionLayer(batch_size=self.batch_size, window=self.window,
                                    bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
                                    images_per_gpu=self.images_per_gpu,
                                    detection_max_instances=100,
                                    detection_min_confidence=0.7,
                                    detection_nms_threshold=0.3, image_shape=self.image_shape,
                                    name='mrcnn_detection')(
            [rpn_rois, classes, mrcnn_bbox])

        detection_boxes = Lambda(lambda x: x[..., :4])(detections)
        mrcnn_mask = build_fpn_mask_graph(detection_boxes, feature_maps[:-1],
                                          num_classes=self.num_classes,
                                          image_shape=self.image_shape)

        inference_model = Model([input_image, anchors],
                                [detections, classes, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                                name='mask_rcnn')
        return inference_model


def get_anchors(image_shape, rpn_anchor_scales, backbone, compute_backbone_shape=None):
    """The function returns a set of anchor
    boxes for a given image size.

    # Returns:
        anchors : Normalized to match the shape of
        the input image [N, (y1, x1, y2, x2)]
    """
    backbone_shapes = compute_backbone_shapes(image_shape, backbone, compute_backbone_shape)
    # Cache anchors and reuse if image shape is the same
    _anchor_cache = {}
    if not tuple(image_shape) in self._anchor_cache:
        # Generate Anchors
        a = generate_pyramid_anchors(rpn_anchor_scales,
                                     [0.5, 1, 2], backbone_shapes,
                                     [4, 8, 16, 32, 64], 1)
        # Normalize coordinates
        _anchor_cache[tuple(image_shape)] = \
            norm_boxes(a, tuple(image_shape[:2]))

    return _anchor_cache[tuple(image_shape)]


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


def upsample_block(y, x, filters, up_sample_name='None', fpn_name='None',
                   fpn_add_name='None'):
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
    apply_filter = convolution_block(x, filters, (1, 1), name=fpn_name)
    p = Add(name=fpn_add_name)([upsample, apply_filter])

    return p


def build_backbone(image_shape, backbone_features, fpn_size, train_bn):
    """Builds ``BACKBONE`` class for mask-RCNN

    # Arguments:
        image_shape: [H,W].
        backbone_features
        fpn_size
        train_bn
    # Returns:
        Model: Keras model
    """
    height, width = image_shape[:2]
    raise_exception(height, width)
    input_image = Input(shape=[None, None, image_shape[2]], name='input_image')

    C2, C3, C4, C5 = get_backbone_features(input_image, backbone_features, train_bn)
    P2, P3, P4, P5, P6 = build_layers(C2, C3, C4, C5, fpn_size)

    model = Model([input_image], [P2, P3, P4, P5, P6], name='mask_rcnn_before')

    return model


def build_layers(C2, C3, C4, C5, fpn_size):
    """Builds `layers for mask-RCNN backbone.

    # Arguments:
        C2, C3, C4, C5: channel sizes
        fpn_size
    # Returns:
        Model layers.
    """
    P5 = convolution_block(C5, fpn_size, (1, 1), name='fpn_c5p5')
    P4 = upsample_block(P5, C4, fpn_size, up_sample_name='fpn_p5upsampled',
                        fpn_name='fpn_c4p4', fpn_add_name='fpn_p4add')
    P3 = upsample_block(P4, C3, fpn_size, up_sample_name='fpn_p4upsampled',
                        fpn_name='fpn_c3p3', fpn_add_name='fpn_p3add')
    P2 = upsample_block(P3, C2, fpn_size, up_sample_name='fpn_p3upsampled',
                        fpn_name='fpn_c2p2', fpn_add_name='fpn_p2add')

    P2 = convolution_block(P2, fpn_size, (3, 3), name='fpn_p2', padd='SAME')
    P3 = convolution_block(P3, fpn_size, (3, 3), name='fpn_p3', padd='SAME')
    P4 = convolution_block(P4, fpn_size, (3, 3), name='fpn_p4', padd='SAME')
    P5 = convolution_block(P5, fpn_size, (3, 3), name='fpn_p5', padd='SAME')
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)
    return P2, P3, P4, P5, P6


def get_backbone_features(image, backbone_features, train_bn):
    """Callable function to get backbone features for mask-RCNN backbone.

    # Arguments:
        image
        backbone_features
        train_BN
    # Returns:
        Model layers.
    """
    if callable(backbone_features):
        _, C2, C3, C4, C5 = backbone_features(image, stage5=True, train_bn=train_bn)
    else:
        _, C2, C3, C4, C5 = get_resnet_features(image, backbone_features, stage5=True,
                                                train_bn=train_bn)

    return C2, C3, C4, C5


def raise_exception(height, width):
    """Raise exception when image is not a multiple of 2.

    # Arguments:
        height, width : size of image

    """
    if height / 2 ** 6 != int(height / 2 ** 6) or width / 2 ** 6 != int(width / 2 ** 6):
        raise Exception('Image size must be dividable by 2 atleast'
                        '6 times')


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


def rpn_model(rpn_anchor_stride, rpn_anchor_ratios, fpn_size, rpn_feature_maps):
    """Build complete region specific network.

    # Arguments:
        fpn_size
        rpn_feature_masks
    # Returns:
        Model output.
    """
    rpn = build_rpn_model(rpn_anchor_stride, len(rpn_anchor_ratios), fpn_size)
    layer_outputs = [rpn([feature]) for feature in rpn_feature_maps]
    names = ['rpn_class_logits', 'rpn_class', 'rpn_bbox']
    outputs = list(zip(*layer_outputs))
    outputs = [Concatenate(axis=1, name=name)(list(output))
               for output, name in zip(outputs, names)]

    return outputs


def get_trainable(model, layer_regex, keras_model=None, indent=0, verbose=1):
    """Build all the layers by selecting the model.

    # Arguments:
        model
        layer_regex
        keras_model
    # Returns:
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
            get_trainable(model, layer_regex=layer_regex, keras_model=layer,
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


def gnd_truth_call(image):
    """Decorator function used to call the norm_boxes_graph function.

    # Arguments
        image: Input image in original form [H, W, C].
        boxes: Bounding box in original form [N, (y1, x1, y2, x2)].
    # Returns
        bounding box: Bounding box in normalised form [N, (y1, x1, y2, x2)].
    """
    shape = tf.shape(image)[1:3]

    def _gnd_truth_call(boxes):
        return norm_boxes_graph(boxes, shape)

    return _gnd_truth_call


def get_ground_truth_values(image_shape, image, mini_mask=False, mini_mask_shape=(56, 56)):
    """Returns groundtruth values needed for network head
    creation of the type required by the model.

    # Arguments:
        input_image: Input image in original form [H, W, C].
    # Returns:
        input class ids: [No. of instances]
        input bounding boxes in normalised form [N, (y1, x1, y2, x2)]
        input groundtruth masks [N, (Shape of Input image/Mini mask shape)]
    """
    class_ids = Input(shape=[None], name='input_gt_class_ids', dtype=tf.int32)
    input_boxes = Input(shape=[None, 4], name='input_gt_boxes', dtype=tf.float32)

    boxes = gnd_truth_call(image)(input_boxes)

    if mini_mask:
        input_gt_masks = Input(
            shape=[mini_mask_shape[0],
                   mini_mask_shape[1], None],
            name="input_gt_masks", dtype=bool)
    else:
        input_gt_masks = Input(
            shape=[image_shape[0], image_shape[1], None],
            name="input_gt_masks", dtype=bool)

    return class_ids, input_boxes, boxes, input_gt_masks


def create_head(backbone_model, ROIs, num_classes, image_shape, train_bn=False):
    """ Creation of region specific network head by calling
    RPN classifier and RPN mask models.

    # Arguments:
        backbone model
        ROIs [No. of ROIs before nms (y1, x1, y2, x2)]
    # Return:
        logits: classifier logits (before softmax)
                [batch, num_rois, NUM_CLASSES]
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                     Deltas to apply to proposal boxes
        Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    feature_maps = backbone_model.output
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
        fpn_classifier_graph(ROIs, feature_maps[:-1], num_classes, image_shape,
                             train_bn=train_bn, fc_layers_size=1024)

    mrcnn_mask = build_fpn_mask_graph(ROIs, feature_maps[:-1], num_classes, image_shape,
                                      train_bn=train_bn)
    output_ROIs = call_ROIs()(ROIs)

    return mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, output_ROIs


def call_ROIs():
    """ Decorator function to call the output ROIs.

    # Returns:
        output ROIs [No. of ROIs before nms (y1, x1, y2, x2)]
    """
    def _call_ROIs(value):
        return value * 1

    return _call_ROIs
