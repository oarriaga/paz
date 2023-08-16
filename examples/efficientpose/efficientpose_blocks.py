import tensorflow as tf
from tensorflow.keras.layers import (
    GroupNormalization, Concatenate, Add, Reshape)
from paz.models.detection.efficientdet.efficientdet_blocks import (
    build_head_conv2D)


def build_pose_estimator_head(middles, num_dims=3):
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
    rotation_outputs = RotationNet(middles)
    rotations = []
    for level_arg in range(len(rotation_outputs)):
        rotation_per_level = rotation_outputs[level_arg]
        rotation_per_level = Reshape((-1, num_dims))(rotation_per_level)
        rotations.append(rotation_per_level)
    rotations = Concatenate(axis=1)(rotations)
    return rotations


def RotationNet(middles, num_iterations=1, num_anchors=9,
                num_filters=64, num_blocks=3, num_dims=3):

    bias_initializer = tf.zeros_initializer()
    num_filters = [num_filters, num_dims * num_anchors]
    rotation_head_outputs, rotations = build_rotation_head(middles, num_blocks,
                                                           num_filters,
                                                           bias_initializer)
    return IterativeRotationSubNet(rotation_head_outputs, rotations,
                                   num_iterations, num_filters,
                                   num_blocks-1, num_dims)


def build_rotation_head(features, num_blocks, num_filters,
                        bias_initializer, gn_groups=4, gn_axis=-1):
    """Builds ClassNet/BoxNet head.

    # Arguments
        middle_features: Tuple. input features.
        num_blocks: Int, number of intermediate layers.
        num_filters: Int, number of intermediate layer filters.
        survival_rate: Float, used by drop connect.
        bias_initializer: Callable, bias initializer.

    # Returns
        head_outputs: List, with head outputs.
    """
    conv_blocks = build_head_conv2D(
        num_blocks, num_filters[0], tf.zeros_initializer())
    final_head_conv = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    rotations, rotation_head_outputs = [], []
    for x in features:
        for block_arg in range(num_blocks):
            x = conv_blocks[block_arg](x)
            x = GroupNormalization(groups=gn_groups, axis=gn_axis)(x)
            x = tf.nn.swish(x)
        rotation = final_head_conv(x)
        rotations.append(rotation)
        rotation_head_outputs.append(x)
    return rotation_head_outputs, rotations


def IterativeRotationSubNet(features, rotations, num_iterations,
                            num_filters, num_blocks, num_dims):
    bias_initializer = tf.zeros_initializer()
    return build_iterative_rotation_head(features, rotations, num_iterations,
                                         num_blocks, num_filters, num_dims,
                                         bias_initializer)


def build_iterative_rotation_head(features, rotations, num_iterations,
                                  num_blocks, num_filters, num_dims,
                                  bias_initializer, gn_groups=4, gn_axis=-1):
    """Builds ClassNet/BoxNet head.

    # Arguments
        middle_features: Tuple. input features.
        num_blocks: Int, number of intermediate layers.
        num_filters: Int, number of intermediate layer filters.
        survival_rate: Float, used by drop connect.
        bias_initializer: Callable, bias initializer.

    # Returns
        head_outputs: List, with head outputs.
    """
    conv_blocks = build_head_conv2D(
        num_blocks, num_filters[0], tf.zeros_initializer())
    final_head_conv = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    iterative_rotation_head_outputs = []
    for x, rotation in zip(features, rotations):
        for _ in range(num_iterations):
            x = Concatenate(axis=-1)([x, rotation])
            for block_arg in range(num_blocks):
                x = conv_blocks[block_arg](x)
                x = GroupNormalization(groups=gn_groups, axis=gn_axis)(x)
                x = tf.nn.swish(x)
            delta_rotation = final_head_conv(x)
            rotation = Add()([rotation, delta_rotation])
        iterative_rotation_head_outputs.append(rotation)
    return iterative_rotation_head_outputs
