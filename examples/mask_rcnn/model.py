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

from .utils import log, get_resnet_features, build_rpn_model
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
        filters: Int. Number of filters
        branch: Tensor to be concatated to the upsamples ``x`` tensor.

    # Returns
        A Keras tensor.
    """
    upsample = UpSampling2D(size=(2, 2), name=up_sample_name)(y)
    conv2d = convolution_block(x, filters, (1, 1), name=fpn_name)
    p = Add(name=fpn_add_name)([upsample, conv2d])

    return p


def build_backbone(image_shape, backbone_features, fpn_size, train_bn):
    height, width = image_shape[:2]
    raise_exception(height, width)
    input_image = Input(shape=[None, None, image_shape[2]], name='input_image')
    C2, C3, C4, C5= get_backbone_features(input_image, backbone_features, train_bn)
    P2, P3, P4, P5, P6 = build_layers(C2, C3, C4, C5, fpn_size)
    model = Model([input_image], [P2, P3, P4, P5, P6], name='mask_rcnn')
    return model

def build_layers(C2, C3, C4, C5, fpn_size):

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
    if callable(backbone_features):
        _, C2, C3, C4, C5 = backbone_features(image, stage5=True, train_bn=train_BN)
    else:
        _, C2, C3, C4, C5 = get_resnet_features(image, backbone_features, stage5=True, train_bn=train_BN)

    return( C2, C3, C4, C5)


def raise_exception( height, width):

    if height / 2 ** 6 != int(height / 2 ** 6) or width / 2 ** 6 != int(width / 2 ** 6):
        raise Exception('Image size must be dividable by 2 atleast'
                        '6 times')


def get_imagenet_weights():
    weight_path = 'https://github.com/fchollet/deep-learning-models/'\
                       'releases/download/v0.2/'\
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    filepath = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_file = get_file(filepath, weight_path, cache_subdir='models',
                            md5_hash='a268eb855778b3df3c7506639542a6af')
    return weights_file


def RPN_model(config, fpn_size, rpn_feature_maps):
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