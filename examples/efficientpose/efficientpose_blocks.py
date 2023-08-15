import tensorflow as tf
from tensorflow.keras.layers import GroupNormalization
from paz.models.detection.efficientdet.efficientdet_blocks import build_head_conv2D
from paz.models.detection.efficientdet.layers import GetDropConnect


def RotationNet(features, num_iterations=1, num_anchors=9, num_filters=64,
                num_blocks=3, num_dims=3):

    bias_initializer = tf.zeros_initializer()
    num_filters = [num_filters, num_dims * num_anchors]
    return build_rotation_head(features, num_blocks, num_filters,
                               bias_initializer)


def build_rotation_head(middle_features, num_blocks, num_filters,
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
    rotation_head_outputs = []
    for x in middle_features:
        for block_arg in range(num_blocks):
            x = conv_blocks[block_arg](x)
            x = GroupNormalization(groups=gn_groups, axis=gn_axis)(x)
            x = tf.nn.swish(x)
        x = final_head_conv(x)
        rotation_head_outputs.append(x)
    return rotation_head_outputs
