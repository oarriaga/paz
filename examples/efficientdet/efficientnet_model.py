import math
import numpy as np
import tensorflow as tf
import itertools
from tensorflow.keras.layers import (DepthwiseConv2D, Conv2D,
                                     BatchNormalization, Input)


def get_efficientnet_params(model_name):
    """Default efficientnet scaling coefficients and
    image name based on model name.
    The value of each model name in the key represents:
    (width_coefficient, depth_coefficient, survival_rate).
    with_coefficient: scaling coefficient for network width.
    depth_coefficient: scaling coefficient for network depth.
    survival_rate: survival rate for final fully connected layers.

    # Arguments
        model_name: String, name of the EfficientNet backbone

    # Returns
        efficientnetparams: Dictionary, parameters corresponding to
        width coefficient, depth coefficient, survival rate
    """
    efficientnet_params = {'efficientnet-b0': (1.0, 1.0, 0.8),
                           'efficientnet-b1': (1.0, 1.1, 0.8),
                           'efficientnet-b2': (1.1, 1.2, 0.7),
                           'efficientnet-b3': (1.2, 1.4, 0.7),
                           'efficientnet-b4': (1.4, 1.8, 0.6),
                           'efficientnet-b5': (1.6, 2.2, 0.6),
                           'efficientnet-b6': (1.8, 2.6, 0.5),
                           'efficientnet-b7': (2.0, 3.1, 0.5),
                           'efficientnet-b8': (2.2, 3.6, 0.5),
                           'efficientnet-l2': (4.3, 5.3, 0.5)}
    return efficientnet_params[model_name]


def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on depth multiplier.

    # Arguments
        filters: Int, filters to be rounded based on depth multiplier.
        with_coefficient: Float, scaling coefficient for network width.
        depth_divisor: Int, multiplier for the depth of the network.
        skip: Bool, skip rounding filters based on multiplier.

    # Returns
        new_filters: Int, rounded filters based on depth multiplier.
    """
    filters = filters * width_coefficient
    min_depth = depth_divisor
    half_depth = depth_divisor / 2
    threshold = int(filters + half_depth) // depth_divisor * depth_divisor
    new_filters = max(min_depth, threshold)
    if new_filters < 0.9 * filters:
        new_filters = new_filters + depth_divisor
    new_filters = int(new_filters)
    return new_filters


def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier.

    # Arguments
        repeats: Int, number of repeats of multiplier blocks.
        depth_coefficient: Float, scaling coefficient for network depth.
        skip: Bool, skip rounding filters based on multiplier.

    # Returns
        new_repeats: Int, repeats of blocks based on multiplier.
    """
    new_repeats = int(math.ceil(depth_coefficient * repeats))
    return new_repeats


def conv_normal_initializer(shape, dtype=None):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.initializers.variance_scaling uses a truncated normal with
    a corrected standard deviation.

    # Arguments
        shape: shape of variable
        dtype: dtype of variable

    # Returns
        an initialization for the variable
    """
    kernel_height, kernel_width, _, output_filters = shape
    fan_output = int(kernel_height * kernel_width * output_filters)
    return tf.random.normal(shape, 0.0, np.sqrt(2.0 / fan_output), dtype)


def get_drop_connect(features, is_training, survival_rate):
    """Drop the entire conv with given survival probability.
    Deep Networks with Stochastic Depth, https://arxiv.org/pdf/1603.09382.pdf

    # Arguments
        features: Tensor, input feature map to undergo
        drop connection.
        is_training: Bool specifying the training phase.
        survival_rate: Float, survival probability to drop
        input convolution features.

    # Returns
        output: Tensor, output feature map after drop connect.
    """
    if not is_training:
        return features
    batch_size = tf.shape(features)[0]
    random_tensor = survival_rate
    random_tensor = random_tensor + tf.random.uniform(
        [batch_size, 1, 1, 1], dtype=features.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = features / survival_rate * binary_tensor
    return output


def get_conv_name(conv_id):
    if not next(conv_id):
        name_appender = ""
    else:
        name_appender = '_' + str(next(conv_id) // 2)
    name = 'conv2d' + name_appender
    return name


def get_batch_norm_name(batch_norm_id):
    if not next(batch_norm_id):
        name_appender = ""
    else:
        name_appender = '_' + str(next(batch_norm_id) // 2)
    name = 'batch_normalization' + name_appender
    return name


def mobile_inverted_residual_bottleneck_block(
        inputs, survival_rate, kernel_size, input_filters, output_filters,
        expand_ratio, strides, squeeze_excite_ratio, name=''):
    """A class of MBConv: Mobile Inverted Residual Bottleneck. As provided in
    the paper: https://arxiv.org/pdf/1801.04381.pdf and
    https://arxiv.org/pdf/1905.11946.pdf
    Initializes a MBConv block.

    # Arguments
        kernel_size: Int, kernel size of the conv block filters.
        num_repeats: Int, number of block repeats.
        input_filters: Int, input filters for the blocks to construct.
        output_filters: Int, output filters for the blocks to construct.
        expand_ratio: Int, ratio to expand the conv block in repeats.
        strides: List, strides in height and weight of conv filter.
        squeeze_excite_ratio: Float, squeeze excite block ratio.
        num_blocks: Int, number of Mobile bottleneck conv blocks.
        name: layer name.
    """
    conv_id = itertools.count(0)
    batch_norm_id = itertools.count(0)
    filters = input_filters * expand_ratio
    if expand_ratio != 1:
        x = Conv2D(filters, 1, padding='same', use_bias=False,
                   kernel_initializer=conv_normal_initializer,
                   name=name + '/' + get_conv_name(conv_id))(inputs)
        x = BatchNormalization(
            -1, 0.99, 1e-3,
            name=name+'/' + get_batch_norm_name(batch_norm_id))(x)
        x = tf.nn.swish(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False,
                        depthwise_initializer=conv_normal_initializer,
                        name=name + '/depthwise_conv2d')(x)
    x = BatchNormalization(
        -1, 0.99, 1e-3,
        name=name + '/' + get_batch_norm_name(batch_norm_id))(x)
    x = tf.nn.swish(x)

    # Squeeze excitation layer
    num_reduced_filters = max(1, int(input_filters * squeeze_excite_ratio))
    se_tensor = tf.reduce_mean(x, [1, 2], keepdims=True)
    se_tensor = Conv2D(num_reduced_filters, 1, padding='same', use_bias=True,
                       kernel_initializer=conv_normal_initializer,
                       name=name + '/se/conv2d')(se_tensor)
    se_tensor = tf.nn.swish(se_tensor)
    se_tensor = Conv2D(filters, 1, padding='same', use_bias=True,
                       kernel_initializer=conv_normal_initializer,
                       name=name + '/se/conv2d_1')(se_tensor)
    se_tensor = tf.sigmoid(se_tensor)
    x = se_tensor * x

    # Output processing
    x = Conv2D(output_filters, 1, padding='same', use_bias=False,
               kernel_initializer=conv_normal_initializer,
               name=name + '/' + get_conv_name(conv_id))(x)
    x = BatchNormalization(
        -1, 0.99, 1e-3,
        name=name + '/' + get_batch_norm_name(batch_norm_id))(x)
    if all(s == 1 for s in strides) and input_filters == output_filters:
        if survival_rate:
            x = get_drop_connect(x, False, survival_rate)
        x = tf.add(x, inputs)
    return x


def EfficientNet(image, model_name, input_shape=(512, 512, 3), depth_divisor=8,
                 squeeze_excite_ratio=0.25, kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
                 num_repeats=[1, 2, 2, 3, 3, 4, 1],
                 input_filters=[32, 16, 24, 40, 80, 112, 192],
                 output_filters=[16, 24, 40, 80, 112, 192, 320],
                 expand_ratios=[1, 6, 6, 6, 6, 6, 6],
                 strides=[[1, 1], [2, 2], [2, 2], [2, 2],
                          [1, 1], [2, 2], [1, 1]]):
    """A class implementing tf.keras.Model for EfficientNet. Base paper:
    https://arxiv.org/pdf/1905.11946.pdf
    Initializes an 'Model' instance.
    # Arguments
        with_coefficient: Float, scaling coefficient for network width.
        depth_coefficient: Float, scaling coefficient for network depth.
        survival_rate: Float, survival of the final fully connected layer
        units.
        name: A string of layer name.
        num_classes: Int, specifying the number of class in the
        output.
        depth_divisor: Int, multiplier for the depth of the network.
        kernel_size: Int, kernel size of the conv block filters.
        num_repeats: Int, number of block repeats.
        input_filters: Int, input filters for the blocks to construct.
        output_filters: Int, output filters for the blocks to construct.
        expand_ratio: Int, ratio to expand the conv block in repeats.
        strides: List, strides in height and weight of conv filter.
        squeeze_excite_ratio: Float, squeeze excite block ratio.

    # Raises
        ValueError: when blocks_args is not specified as list.
    """
    block_id = itertools.count(0)
    features = []
    (width_coefficient, depth_coefficient,
     survival_rate) = get_efficientnet_params(model_name)

    image = Input(tensor=image, shape=input_shape)
    filters = round_filters(input_filters[0], width_coefficient, depth_divisor)
    x = Conv2D(filters, [3, 3], [2, 2], 'same', 'channels_last', (1, 1), 1,
               None, False, conv_normal_initializer,
               name=model_name + '/stem/conv2d')(image)
    x = BatchNormalization(-1, 0.99, 1e-3,
                           name=model_name + '/stem/batch_normalization')(x)
    x = tf.nn.swish(x)
    for block_num in range(len(kernel_sizes)):
        assert num_repeats[block_num] > 0
        in_filter = round_filters(input_filters[block_num],
                                  width_coefficient, depth_divisor)
        out_filter = round_filters(output_filters[block_num],
                                   width_coefficient, depth_divisor)
        repeats = round_repeats(num_repeats[block_num], depth_coefficient)
        x = mobile_inverted_residual_bottleneck_block(
            x, survival_rate, kernel_sizes[block_num], in_filter, out_filter,
            expand_ratios[block_num], strides[block_num], squeeze_excite_ratio,
            model_name + '/blocks_%d' % next(block_id))
        if repeats > 1:
            for _ in range(repeats - 1):
                x = mobile_inverted_residual_bottleneck_block(
                    x, survival_rate, kernel_sizes[block_num], out_filter,
                    out_filter, expand_ratios[block_num], [1, 1],
                    squeeze_excite_ratio,
                    model_name + '/blocks_%d' % next(block_id))
        if (block_num < len(kernel_sizes) - 1 and
                strides[block_num + 1][0] == 2):
            features.append(x)
        elif block_num == len(kernel_sizes) - 1:
            features.append(x)
    return features
