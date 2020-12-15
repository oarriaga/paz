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
