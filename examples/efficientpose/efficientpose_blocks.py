import tensorflow as tf
from tensorflow.keras.layers import (
    GroupNormalization, Concatenate, Add, Reshape)
from paz.models.detection.efficientdet.efficientdet_blocks import (
    build_head_conv2D)


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
        outputs = Reshape((-1, num_dims))(rotation)
        iterative_rotation_head_outputs.append(outputs)
    return Concatenate(axis=1)(iterative_rotation_head_outputs)
