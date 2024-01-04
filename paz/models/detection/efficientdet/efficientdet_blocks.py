import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, Concatenate, Reshape
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Flatten,
                                     MaxPooling2D, SeparableConv2D,
                                     UpSampling2D, GroupNormalization)
from .layers import FuseFeature, GetDropConnect


def build_detector_head(middles, num_classes, num_dims, aspect_ratios,
                        num_scales, FPN_num_filters, box_class_repeats,
                        survival_rate):
    """Builds EfficientDet object detector's head.
    The built head includes ClassNet and BoxNet for classification and
    regression respectively.

    # Arguments
        middles: List, BiFPN layer output.
        num_classes: Int, number of object classes.
        num_dims: Int, number of output dimensions to regress.
        aspect_ratios: List, anchor boxes aspect ratios.
        num_scales: Int, number of anchor box scales.
        FPN_num_filters: Int, number of FPN filters.
        box_class_repeats: Int, Number of regression
            and classification blocks.
        survival_rate: Float, used in drop connect.

    # Returns
        outputs: Tensor of shape `[num_boxes, num_classes+num_dims]`
    """
    num_anchors = len(aspect_ratios) * num_scales
    args = (middles, num_anchors, FPN_num_filters,
            box_class_repeats, survival_rate)
    _, class_outputs = ClassNet(*args, num_classes)
    class_outputs = [Flatten()(class_output) for class_output in class_outputs]
    _, boxes_outputs = BoxesNet(*args, num_dims)
    boxes_outputs = [Flatten()(boxes_output) for boxes_output in boxes_outputs]
    classes = Concatenate(axis=1)(class_outputs)
    regressions = Concatenate(axis=1)(boxes_outputs)
    num_boxes = K.int_shape(regressions)[-1] // num_dims
    classes = Reshape((num_boxes, num_classes))(classes)
    classes = Activation('softmax')(classes)
    regressions = Reshape((num_boxes, num_dims))(regressions)
    outputs = Concatenate(axis=2, name='boxes')([regressions, classes])
    return outputs


def ClassNet(features, num_anchors=9, num_filters=32, num_blocks=4,
             survival_rate=None, num_classes=90):
    """Initializes ClassNet.

    # Arguments
        features: List, input features.
        num_anchors: Int, number of anchors.
        num_filters: Int, number of intermediate layer filters.
        num_blocks: Int, Number of intermediate layers.
        survival_rate: Float, used in drop connect.
        num_classes: Int, number of object classes.

    # Returns
        class_outputs: List, ClassNet outputs per level.
    """
    bias_initializer = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
    num_filters = [num_filters, num_classes * num_anchors]
    return build_head(features, num_blocks, num_filters,
                      bias_initializer, survival_rate)


def BoxesNet(features, num_anchors=9, num_filters=32, num_blocks=4,
             survival_rate=None, num_dims=4):
    """Initializes BoxNet.

    # Arguments
        features: List, input features.
        num_anchors: Int, number of anchors.
        num_filters: Int, number of intermediate layer filters.
        num_blocks: Int, Number of intermediate layers.
        survival_rate: Float, used by drop connect.
        num_dims: Int, number of output dimensions to regress.

    # Returns
        boxes_outputs: List, BoxNet outputs per level.
    """
    bias_initializer = tf.zeros_initializer()
    num_filters = [num_filters, num_dims * num_anchors]
    return build_head(features, num_blocks, num_filters,
                      bias_initializer, survival_rate)


def build_head(middle_features, num_blocks, num_filters,
               bias_initializer, survival_rate, normalization='batch'):
    """Builds ClassNet/BoxNet head.

    # Arguments
        middle_features: Tuple. input features.
        num_blocks: Int, number of intermediate layers.
        num_filters: Int, number of intermediate layer filters.
        bias_initializer: Callable, bias initializer.
        survival_rate: Float, used by drop connect.

    # Returns
        head_outputs: List, with head outputs.
    """
    conv_blocks = build_head_conv2D(
        num_blocks, num_filters[0], tf.zeros_initializer())
    final_head_conv = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    pre_head_outputs, head_outputs = [], []

    if normalization == 'batch':
        normalizer = BatchNormalization
        args = ()

    elif normalization == 'group':
        normalizer = GroupNormalization
        args = (int(num_filters[0] / 16), )

    for x in middle_features:
        for block_arg in range(num_blocks):
            x = conv_blocks[block_arg](x)
            x = normalizer(*args)(x)
            x = tf.nn.swish(x)
            if block_arg > 0 and survival_rate:
                x = x + GetDropConnect(survival_rate=survival_rate)(x)
        pre_head_outputs.append(x)
        x = final_head_conv(x)
        head_outputs.append(x)
    return [pre_head_outputs, head_outputs]


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
    args_1 = (num_filters, 3, (1, 1), 'same', 'channels_last', (1, 1),
              1, None, True)
    for _ in range(num_blocks):
        args_2 = (tf.initializers.variance_scaling(),
                  tf.initializers.variance_scaling(), bias_initializer)
        conv_blocks.append(SeparableConv2D(*args_1, *args_2))
    return conv_blocks


def EfficientNet_to_BiFPN(branches, num_filters):
    """Preprocess EfficientNet branches prior to feeding BiFPN block.
    The branches generated by the EfficientNet backbone consists of
    features P1, P2, P3, P4, and P5. However, the BiFPN block requires
    features P3, P4, P5, P6, and P7. This function generates features
    P3 to P7 from EfficientNet branches that can be fed to the BiFPN
    block.

    # Arguments
        branches: List, EfficientNet feature maps.
        num_filters: Int, number of intermediate layer filters.

    # Returns
        branches, middles, skips: List, extended branch
            and preprocessed feature maps.
    """
    branches = extend_branch(branches, num_filters)
    P3, P4, P5, P6, P7 = branches
    P3_middle = conv_batchnorm_block(P3, num_filters)
    P4_middle = conv_batchnorm_block(P4, num_filters)
    P5_middle = conv_batchnorm_block(P5, num_filters)
    middles = [P3_middle, P4_middle, P5_middle, P6, P7]

    P4_skip = conv_batchnorm_block(P4, num_filters)
    P5_skip = conv_batchnorm_block(P5, num_filters)
    skips = [None, P4_skip, P5_skip, P6, None]
    return [branches, middles, skips]


def extend_branch(branches, num_filters):
    """Extends branches to comply with BiFPN.
    The input branchs includes features P1-P5. This function extends the
    EfficientNet backbone generated branch. The extended branch contains
    features P3-P7.

    # Arguments
        branches: List, EfficientNet feature maps.
        num_filters: Int, number of intermediate layer filters.

    # Returns
        middles, skips: List, modified branch.
    """
    _, _, P3, P4, P5 = branches
    P6, P7 = build_branch(P5, num_filters)
    branches_extended = [P3, P4, P5, P6, P7]
    return branches_extended


def build_branch(P5, num_filters):
    """Builds feature maps P6 and P7 from P5.

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


def BiFPN(middles, skips, num_filters, fusion):
    """BiFPN block.
    BiFPN stands for Bidirectional Feature Pyramid Network.

    # Arguments
        middles: List, BiFPN layer output.
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


def node_BiFPN(up, middle, down, skip, num_filters, fusion):
    """Simulates a single node of BiFPN block.

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
