import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Layer,
                                     MaxPooling2D, SeparableConv2D,
                                     UpSampling2D)

from utils import GetDropConnect


def ClassNet(features, num_anchors=9, num_filters=32, min_level=3, max_level=7,
             repeats=4, survival_rate=None, num_classes=90, return_base=False):
    """Object class prediction network. Initialize the ClassNet.

    # Arguments
        features: List, feature to be processed by ClassNet head.
        num_classes: Integer. Number of classes.
        num_anchors: Integer. Number of anchors.
        num_filters: Integer. Number of filters for intermediate
            layers.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        repeats: Integer. Number of intermediate layers.
        survival_rate: Float. If a value is set then drop connect will
            be used.
        return_base: Bool. Build the base feature network only.
            Excluding final class head.
        name: String indicating the name of this layer.

    # Returns
        box_outputs: List, BoxNet output offsets for each level.
    """
    args = (repeats, num_filters, min_level, max_level, num_classes,
            num_anchors, features, survival_rate, return_base)
    class_outputs = build_predictionnet(*args, build_classnet=True)
    return class_outputs


def BoxNet(features, num_anchors=9, num_filters=32, min_level=3,
           max_level=7, repeats=4, survival_rate=None, return_base=False):
    """Initialize the BoxNet.

    # Arguments
        features: List, feature to be processed by BoxNet head.
        num_anchors: Integer. Number of anchors.
        num_filters: Integer. Number of filters for intermediate
            layers.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        repeats: Integer. Number of intermediate layers.
        survival_rate: Float. If a value is set then drop connect
            will be used.
        return_base: Bool. Build the base feature network only.
            Excluding final class head.
        name: String indicating the name of this layer.

    # Returns
        box_outputs: List, BoxNet output offsets for each level.
    """
    args = (repeats, num_filters, min_level, max_level, None,
            num_anchors, features, survival_rate, return_base)
    box_outputs = build_predictionnet(*args, build_classnet=False)
    return box_outputs


class FuseFeature(Layer):
    def __init__(self, fusion, **kwargs):
        super().__init__(**kwargs)
        self.fusion = fusion
        if fusion == 'fast':
            self.fuse_method = self._fuse_fast
        elif fusion == 'sum':
            self.fuse_method = self._fuse_sum
        else:
            raise ValueError('FPN weight fusion is not defined')

    def build(self, input_shape):
        num_in = len(input_shape)
        args = (self.name, (num_in,), tf.float32,
                tf.keras.initializers.constant(1 / num_in))
        self.w = self.add_weight(*args, trainable=True)

    def call(self, inputs, fusion):
        """
        # Arguments
        inputs: Tensor. Features to be fused.
        fusion: String representing the feature fusion method.

        # Returns
        x: feature after combining by the feature fusion method in
           BiFPN.
        """
        inputs = [input for input in inputs if input is not None]
        return self.fuse_method(inputs)

    def _fuse_fast(self, inputs):
        w = tf.keras.activations.relu(self.w)

        pre_activations = []
        for input_arg in range(len(inputs)):
            pre_activations.append(w[input_arg] * inputs[input_arg])
        x = tf.reduce_sum(pre_activations, 0)
        x = x / (tf.reduce_sum(w) + 0.0001)
        return x

    def _fuse_sum(self, inputs):
        x = inputs[0]
        for node in inputs[1:]:
            x = x + node
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({'fusion': self.fusion})
        return config


def conv2D_layer(num_filters, kernel_size, padding, activation,
                 bias_initializer):
    """Gets the conv2D layer in ClassNet class.
    # Arguments
        num_filters: Integer. Number of intermediate layers.
        kernel_size: Integer. Kernel size of conv layer.
        padding: String. Padding for conv layer.
        activation: String. Activation function.
        name: String. Name of conv layer.
        bias_initializer: String or TF Function. Bias initialization.

    # Returns
        conv2D_layer: TF conv layer.
    """
    args = (num_filters, kernel_size, (1, 1), padding, 'channels_last',
            (1, 1), 1, activation, True, tf.initializers.variance_scaling(),
            tf.initializers.variance_scaling(), bias_initializer)
    conv2D_layer = SeparableConv2D(*args)
    return conv2D_layer


def build_predictionnet(repeats, num_filters, min_level, max_level,
                        num_classes, num_anchors, features, survival_rate,
                        return_base, build_classnet):
    """Builds Prediction Net part of the Efficientdet

    # Arguments
        repeats: List, feature to be processed by PredictionNet head.
        num_filters: Integer. Number of filters for intermediate
            layers.
        with_separable_conv: Bool.
        name: String indicating the name of this layer.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        num_classes: Integer. Number of classes.
        num_anchors: Integer. Number of anchors.
        features: Tuple. Input features for PredictionNet
        survival_rate: Float. If a value is set then drop connect
            will be used.
        return_base: Bool. Build the base feature network only.
            Excluding final class head.
        build_classnet: Bool.

    # Returns
        predictor_outputs: List. Output of PredictionNet block.
    """
    conv_blocks = build_predictionnet_conv_blocks(repeats, num_filters)

    args = (repeats, min_level, max_level)
    batchnorms = build_predictionnet_batchnorm_blocks(*args)

    if build_classnet:
        bias_initializer = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        num_filters = num_classes * num_anchors
        num_levels = max_level - min_level + 1
    else:
        bias_initializer = tf.zeros_initializer()
        num_filters = 4 * num_anchors
        num_levels = len(features)

    classes = conv2D_layer(num_filters, 3, 'same', None, bias_initializer)

    predictor_outputs = []
    for level_id in range(num_levels):
        args = (features, level_id, repeats, conv_blocks, batchnorms,
                survival_rate, return_base, classes)
        level_feature_map = propagate_forward_predictionnet(*args)
        predictor_outputs.append(level_feature_map)
    return predictor_outputs


def build_predictionnet_conv_blocks(repeats, num_filters):
    """Builds convolutional blocks for PredictionNet

    # Arguments
        repeats: List, feature to be processed by PredictionNet head.
        num_filters: Integer. Number of filters for intermediate
            layers.
        name: String indicating the name of this layer.
        is_classnet: Bool.

    # Returns
        conv_blocks: List. Convolutional blocks for PredictionNet.
    """
    conv_blocks = []
    for _ in range(repeats):
        args = (num_filters, 3, 'same', None, tf.zeros_initializer())
        conv_blocks.append(conv2D_layer(*args))
    return conv_blocks


def build_predictionnet_batchnorm_blocks(repeats, min_level, max_level):
    """Builds batch normalization blocks for PredictionNet

    # Arguments
        repeats: List, feature to be processed by PredictionNet head.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        name: String indicating the name of this layer.
        is_classnet: Bool.

    # Returns
        batchnorms: List. Batch normalization blocks for PredictionNet.
    """
    batchnorms = []
    for _ in range(repeats):
        batchnorm_per_level = []
        for _ in range(min_level, max_level + 1):
            batchnorm_per_level.append(BatchNormalization())
        batchnorms.append(batchnorm_per_level)
    return batchnorms


def propagate_forward_predictionnet(features, level_id, repeats, conv_blocks,
                                    batchnorms, survival_rate,
                                    return_base, output_candidates):
    """Propagates features through PredictionNet block.

    # Arguments
        features: Tuple. Input features for PredictionNet
        level_id: Int. The index of feature level.
        repeats: List, feature to be processed by PredictionNet head.
        conv_blocks: List. Convolutional blocks for PredictionNet.
        batchnorms: List. Batch normalization blocks for PredictionNet.
        survival_rate: Float. If a value is set then drop connect will
            be used.
        return_base: Bool. Build the base feature network only.
            Excluding final class head.
        output_candidates: Tensor. PredictionNet output for each level.

    # Returns
        level_feature: List. Batch normalization blocks for
                       PredictionNet.
    """
    level_feature_map = features[level_id]
    for repeat_args in range(repeats):
        original_level_feature_map = level_feature_map
        level_feature_map = conv_blocks[repeat_args](level_feature_map)
        level_feature_map = batchnorms[repeat_args][level_id](
            level_feature_map)
        level_feature_map = tf.nn.swish(level_feature_map)
        if repeat_args > 0 and survival_rate:
            level_feature_map = GetDropConnect(
                survival_rate=survival_rate)(level_feature_map)
            level_feature_map = level_feature_map + original_level_feature_map

    if return_base:
        return level_feature_map
    else:
        return output_candidates(level_feature_map)


def efficientnet_to_BiFPN(branches, num_filters):
    """Modifies the branches produced by EfficientNet backbone such that
    it can be fed into the BiFPN blocks. This modification includes
    generating feature maps P6 and P7 and preprocessing by applying
    2D convolution and batch normalization.

    # Arguments
        branches: List. It is a list of tensors containing the feature
            maps from the output layers of EfficientNet backbone.
        num_filters: Integer. Number of filters for intermediate
            layers.

    # Returns
        middles skips: List, of tensors or feature maps obtained as a
            result of preprocessing EfficientNet feature maps.
    """
    _, _, P3, P4, P5 = branches
    P6, P7 = build_branch(P5, num_filters)
    branches_extended = [P3, P4, P5, P6, P7]
    middles, skips = preprocess_node(branches_extended, num_filters)
    return [middles, skips]


def build_branch(P5, num_filters):
    """Computes feature maps P6 and P7 from P5.

    # Arguments
        P5: Tensor. Output feature map obtained from fifth layer of
            EfficientNet backbone.
        num_filters: Integer. Number of filters for intermediate
            layers.

    # Returns
        P6, P7: List, of tensors or feature maps obtained from sixth
            and seventh layer of EfficientNet backbone.
    """
    P6 = conv_batchnorm_block(P5, num_filters)
    P6 = MaxPooling2D(3, 2, 'same')(P6)
    P7 = MaxPooling2D(3, 2, 'same')(P6)
    return [P6, P7]


def preprocess_node(branches, num_filters):
    """Preprocesses the feature maps obtained from EfficientNet backbone
    by applying 2D convolution and bath normalization such that it can
    be fed into the BiFPN block.

    # Arguments
        branches: List. It is a list of tensors containing the feature
            maps from the output layers of EfficientNet backbone.
        num_filters: Integer. Number of filters for intermediate
            layers.

    # Returns
        middles, skips: List, of tensors or feature maps obtained after
            preprocessing.
    """
    P3, P4, P5, P6, P7 = branches
    P3_middle = conv_batchnorm_block(P3, num_filters)
    P4_middle = conv_batchnorm_block(P4, num_filters)
    P5_middle = conv_batchnorm_block(P5, num_filters)
    middles = [P3_middle, P4_middle, P5_middle, P6, P7]

    P4_skip = conv_batchnorm_block(P4, num_filters)
    P5_skip = conv_batchnorm_block(P5, num_filters)
    skips = [None, P4_skip, P5_skip, P6, None]
    return [middles, skips]


def conv_batchnorm_block(x, num_filters):
    """Builds 2D convolution and batch normalization layers.

    # Arguments
        x: Tensor. Input feature map.
        num_filters: Integer. Number of filters for intermediate
            layers.

    # Returns
        x: Tensor. Feature map obtained as output after applying 2D
            convolution and batch normalization.
    """
    x = Conv2D(num_filters, 1, 1, 'same')(x)
    x = BatchNormalization()(x)
    return x


def node_BiFPN(up, middle, down, skip, num_filters, fusion):
    """Implements the functionality of nodes in the BiFPN block.

    # Arguments
        up: Tensor, upsampled feature map.
        middle: Tensor, Output feature map from preprocess/BiFPN node.
        down: Tensor, Downsampled feature map.
        skip: Tensor, Skip feature map.
        num_filters: Integer. Number of filters for intermediate
            layers.
        fusion: String representing the feature fusion method.

    # Returns
        middle: Tensor. Feature map obtained as output from the BiFPN
            node.
    """
    is_layer_1 = down is None
    if is_layer_1:
        to_fuse = [middle, up]
    else:
        to_fuse = [middle, down] if skip is None else [skip, middle, down]
    middle = FuseFeature(fusion=fusion)(to_fuse, fusion)
    middle = tf.nn.swish(middle)
    middle = SeparableConv2D(num_filters, 3, 1, 'same', use_bias=True)(middle)
    middle = BatchNormalization()(middle)
    return middle


def BiFPN(middles, skips, num_filters, fusion):
    """BiFPN block.

    # Arguments
        middles: Tensor. Feature map obtained as output from the
            preprocess/BiFPN node.
        skips: Tensor. Skip feature map obtained as output from the
            preprocess/BiFPN node.
        num_filters: Integer. Number of filters for intermediate
            layers.
        fusion: String representing the feature fusion method.

    # Returns
        middles, middles: List, feature maps obtained as output from
            BiFPN block.
    """
    P3_middle, P4_middle, P5_middle, P6_middle, P7_middle = middles
    _, P4_skip, P5_skip, P6_skip, _ = skips

    # Downpropagation ---------------------------------------------------------
    P7_up = UpSampling2D()(P7_middle)
    P6_TD = node_BiFPN(P7_up, P6_middle, None, None, num_filters, fusion)
    P6_up = UpSampling2D()(P6_TD)
    P5_TD = node_BiFPN(P6_up, P5_middle, None, None, num_filters, fusion)
    P5_up = UpSampling2D()(P5_TD)
    P4_TD = node_BiFPN(P5_up, P4_middle, None, None, num_filters, fusion)
    P4_up = UpSampling2D()(P4_TD)
    P3_out = node_BiFPN(P4_up, P3_middle, None, None, num_filters, fusion)

    # Upward propagation ------------------------------------------------------
    P3_down = MaxPooling2D(3, 2, 'same')(P3_out)
    P4_out = node_BiFPN(None, P4_TD, P3_down, P4_skip, num_filters, fusion)
    P4_down = MaxPooling2D(3, 2, 'same')(P4_out)
    P5_out = node_BiFPN(None, P5_TD, P4_down, P5_skip, num_filters, fusion)
    P5_down = MaxPooling2D(3, 2, 'same')(P5_out)
    P6_out = node_BiFPN(None, P6_TD, P5_down, P6_skip, num_filters, fusion)
    P6_down = MaxPooling2D(3, 2, 'same')(P6_out)
    P7_out = node_BiFPN(None, P7_middle, P6_down, None, num_filters, fusion)

    middles = [P3_out, P4_out, P5_out, P6_out, P7_out]
    return [middles, middles]
