import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Flatten,
                                     MaxPooling2D, SeparableConv2D,
                                     UpSampling2D)
from layers import FuseFeature, GetDropConnect


def ClassNet(features, num_anchors=9, num_filters=32, num_blocks=4,
             survival_rate=None, return_base=False, num_classes=90):
    """Initializes ClassNet.

    # Arguments
        features: List, input features.
        num_anchors: Int, number of anchors.
        num_filters: Int, number of intermediate layer filters.
        num_blocks: Int, Number of intermediate layers.
        survival_rate: Float, used in drop connect.
        return_base: Bool, to build only base feature network.
        num_classes: Int, number of object classes.

    # Returns
        class_outputs: List, ClassNet outputs per level.
    """
    bias_initializer = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
    num_filters = [num_filters, num_classes * num_anchors]
    return build_head(features, num_blocks, num_filters, survival_rate,
                      return_base, bias_initializer)


def BoxesNet(features, num_anchors=9, num_filters=32, num_blocks=4,
             survival_rate=None, return_base=False, num_dims=4):
    """Initializes BoxNet.

    # Arguments
        features: List, input features.
        num_anchors: Int, number of anchors.
        num_filters: Int, number of intermediate layer filters.
        num_blocks: Int, Number of intermediate layers.
        survival_rate: Float, used by drop connect.
        return_base: Bool, to build only base feature network.
        num_dims: Int, number of output dimensions to regress.

    # Returns
        boxes_outputs: List, BoxNet outputs per level.
    """
    bias_initializer = tf.zeros_initializer()
    num_filters = [num_filters, num_dims * num_anchors]
    return build_head(features, num_blocks, num_filters, survival_rate,
                      return_base, bias_initializer)


def build_head(middle_features, num_blocks, num_filters,
               survival_rate, return_base, bias_initializer):
    """Builds head.

    # Arguments
        middle_features: Tuple. input features.
        num_blocks: Int, number of intermediate layers.
        num_filters: Int, number of intermediate layer filters.
        survival_rate: Float, used by drop connect.
        return_base: Bool, to build only base feature network.
        bias_initializer: Callable, bias initializer.

    # Returns
        head_outputs: List, with head outputs.
    """
    conv_blocks = build_head_conv2D(
        num_blocks, num_filters[0], tf.zeros_initializer())
    final_head_conv = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    head_outputs = []
    for x in middle_features:
        for block_arg in range(num_blocks):
            x = conv_blocks[block_arg](x)
            x = BatchNormalization()(x)
            x = tf.nn.swish(x)
            if block_arg > 0 and survival_rate:
                x = x + GetDropConnect(survival_rate=survival_rate)(x)
        if not return_base:
            x = final_head_conv(x)
            x = Flatten()(x)
        head_outputs.append(x)
    return head_outputs


def build_head_conv2D(num_blocks, num_filters, bias_initializer):
    """Builds head convolutional blocks.

    # Arguments
        num_blocks: Int, number of intermediate layers.
        num_filters: Int, number of intermediate layer filters.
        bias_initializer: Callable, bias initializer.

    # Returns
        conv_blocks: List, head convolutional blocks.
    """
    conv_blocks = []
    for _ in range(num_blocks):
        args = (num_filters, 3, (1, 1), 'same', 'channels_last', (1, 1),
                1, None, True, tf.initializers.variance_scaling(),
                tf.initializers.variance_scaling(), bias_initializer)
        conv_blocks.append(SeparableConv2D(*args))
    return conv_blocks


def EfficientNet_to_BiFPN(branches, num_filters):
    """Modifies the branches to comply with BiFPN.

    # Arguments
        branches: List, EfficientNet feature maps.
        num_filters: Int, number of intermediate layer filters.

    # Returns
        middles, skips: List, modified branch.
    """
    _, _, P3, P4, P5 = branches
    P6, P7 = build_branch(P5, num_filters)
    branches_extended = [P3, P4, P5, P6, P7]
    middles, skips = preprocess_node(branches_extended, num_filters)
    return [middles, skips]


def build_branch(P5, num_filters):
    """Builds feature maps P6 and P7.

    # Arguments
        P5: Tensor of shape `(batch_size, 16, 16, 320)`,
            EfficientNet's 5th layer output.
        num_filters: Int, number of intermediate layer filters.

    # Returns
        P6, P7: List, EfficientNet's 6th and 7th layer output.
    """
    P6 = conv_batchnorm_block(P5, num_filters)
    P6 = MaxPooling2D(3, 2, 'same')(P6)
    P7 = MaxPooling2D(3, 2, 'same')(P6)
    return [P6, P7]


def preprocess_node(branches, num_filters):
    """Preprocess EfficientNet features.

    # Arguments
        branches: List, EfficientNet feature maps.
        num_filters: Int, number of intermediate layer filters.

    # Returns
        middles, skips: List, preprocessed feature maps.
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
        x: Tensor, input feature map.
        num_filters: Int, number of intermediate layer filters.

    # Returns
        x: Tensor. Feature after convolution and batch normalization.
    """
    x = Conv2D(num_filters, 1, 1, 'same')(x)
    x = BatchNormalization()(x)
    return x


def node_BiFPN(up, middle, down, skip, num_filters, fusion):
    """Simulates BiFPN block's node.

    # Arguments
        up: Tensor, upsampled feature map.
        middle: Tensor, preprocessed feature map.
        down: Tensor, downsampled feature map.
        skip: Tensor, skip feature map.
        num_filters: Int, number of intermediate layer filters.
        fusion: Str, feature fusion method.

    # Returns
        middle: Tensor, BiFPN node output.
    """
    is_layer_one = down is None
    if is_layer_one:
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
        middles: List, BiFPN node output.
        skips: List, skip feature map from BiFPN node.
        num_filters: Int, number of intermediate layer filters.
        fusion: Str, feature fusion method.

    # Returns
        middles, middles: List, BiFPN block output.
    """
    P3_middle, P4_middle, P5_middle, P6_middle, P7_middle = middles
    _, P4_skip, P5_skip, P6_skip, _ = skips

    # Downpropagation ---------------------------------------------------------
    args = (num_filters, fusion)
    P7_up = UpSampling2D()(P7_middle)
    P6_top_down = node_BiFPN(P7_up, P6_middle, None, None, *args)
    P6_up = UpSampling2D()(P6_top_down)
    P5_top_down = node_BiFPN(P6_up, P5_middle, None, None, *args)
    P5_up = UpSampling2D()(P5_top_down)
    P4_top_down = node_BiFPN(P5_up, P4_middle, None, None, *args)
    P4_up = UpSampling2D()(P4_top_down)
    P3_out = node_BiFPN(P4_up, P3_middle, None, None, *args)

    # Upward propagation ------------------------------------------------------
    P3_down = MaxPooling2D(3, 2, 'same')(P3_out)
    P4_out = node_BiFPN(None, P4_top_down, P3_down, P4_skip, *args)
    P4_down = MaxPooling2D(3, 2, 'same')(P4_out)
    P5_out = node_BiFPN(None, P5_top_down, P4_down, P5_skip, *args)
    P5_down = MaxPooling2D(3, 2, 'same')(P5_out)
    P6_out = node_BiFPN(None, P6_top_down, P5_down, P6_skip, *args)
    P6_down = MaxPooling2D(3, 2, 'same')(P6_out)
    P7_out = node_BiFPN(None, P7_middle, P6_down, None, *args)

    middles = [P3_out, P4_out, P5_out, P6_out, P7_out]
    return [middles, middles]
