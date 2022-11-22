import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     DepthwiseConv2D, Input)


def get_scaling_coefficients(model_name):
    """Default efficientnet scaling coefficients and
    image name based on model name.
    The value of each model name in the key represents:
    (W_coefficient, D_coefficient, survival_rate).
    with_coefficient: scaling coefficient for network width.
    D_coefficient: scaling coefficient for network depth.
    survival_rate: survival rate for final fully connected layers.

    # Arguments
        model_name: String, name of the EfficientNet backbone

    # Returns
        efficientnetparams: Dictionary, parameters corresponding to
            width coefficient, depth coefficient, survival rate
    """
    scaling_coefficients = {'efficientnet-b0': (1.0, 1.0, 0.8),
                            'efficientnet-b1': (1.0, 1.1, 0.8),
                            'efficientnet-b2': (1.1, 1.2, 0.7),
                            'efficientnet-b3': (1.2, 1.4, 0.7),
                            'efficientnet-b4': (1.4, 1.8, 0.6),
                            'efficientnet-b5': (1.6, 2.2, 0.6),
                            'efficientnet-b6': (1.8, 2.6, 0.5),
                            'efficientnet-b7': (2.0, 3.1, 0.5),
                            'efficientnet-b8': (2.2, 3.6, 0.5),
                            'efficientnet-l2': (4.3, 5.3, 0.5)}
    return scaling_coefficients[model_name]


def round_filters(filters, W_coefficient, D_divisor):
    """Round number of filters based on depth multiplier.

    # Arguments
        filters: Int, filters to be rounded based on depth multiplier.
        W_coefficient: Float, scaling coefficient for network width.
        D_divisor: Int, multiplier for the depth of the network.

    # Returns
        new_filters: Int, rounded filters based on depth multiplier.
    """
    filters = filters * W_coefficient
    min_D = D_divisor
    half_D = D_divisor / 2
    threshold = (int(filters + half_D) // D_divisor) * D_divisor
    new_filters = int(max(min_D, threshold))
    if new_filters < 0.9 * filters:
        new_filters = int(new_filters + D_divisor)
    return new_filters


def round_repeats(repeats, D_coefficient):
    """Round number of repeat blocks based on depth multiplier.

    # Arguments
        repeats: Int, number of repeats of multiplier blocks.
        D_coefficient: Float, scaling coefficient for network depth.

    # Returns
        new_repeats: Int, repeats of blocks based on multiplier.
    """
    new_repeats = int(math.ceil(D_coefficient * repeats))
    return new_repeats


def conv_normal_initializer(shape, dtype=None):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an
    uncorrected standard deviation, whereas here we use a normal
    distribution. Similarly, tf.initializers.variance_scaling uses a
    truncated normal with a corrected standard deviation.

    # Arguments
        shape: shape of variable
        dtype: dtype of variable

    # Returns
        an initialization for the variable
    """
    kernel_H, kernel_W, _, outro_filters = shape
    fan_output = int(kernel_H * kernel_W * outro_filters)
    return tf.random.normal(shape, 0.0, np.sqrt(2.0 / fan_output), dtype)


def get_drop_connect(x, is_training, survival_rate):
    """Drop the entire conv with given survival probability.
    Deep Networks with Stochastic Depth,
    https://arxiv.org/pdf/1603.09382.pdf

    # Arguments
        x: Tensor, input feature map to undergo drop connection.
        is_training: Bool specifying the training phase.
        survival_rate: Float, survival probability to drop.

    # Returns
        output: Tensor, output feature map after drop connect.
    """
    if not is_training:
        return x
    batch_size = tf.shape(x)[0]
    random_tensor = survival_rate
    random_tensor = random_tensor + tf.random.uniform(
        [batch_size, 1, 1, 1], dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = x / survival_rate * binary_tensor
    return output


def mobile_inverted_residual_bottleneck_block(
        inputs, survival_rate, kernel_size, intro_filters, outro_filters,
        expand_ratio, strides, squeeze_excite_ratio):
    """A class of MBConv: Mobile Inverted Residual Bottleneck. As
    provided in the paper: https://arxiv.org/pdf/1801.04381.pdf and
    https://arxiv.org/pdf/1905.11946.pdf
    Initializes a MBConv block.

    # Arguments
        kernel_size: Int, kernel size of the conv block filters.
        repeats: Int, number of block repeats.
        intro_filters: Int, input filters for the blocks to construct.
        outro_filters: Int, output filters for the blocks to construct.
        expand_ratio: Int, ratio to expand the conv block in repeats.
        strides: List, strides in height and weight of conv filter.
        squeeze_excite_ratio: Float, squeeze excite block ratio.
        num_blocks: Int, number of Mobile bottleneck conv blocks.

    # Returns
        x: Tensor, output features.
    """
    filters = intro_filters * expand_ratio
    x = build_MBblock_input_layer(
        expand_ratio, filters, conv_normal_initializer, inputs)

    x = build_MBblock_depth_wise_conv_layer(
        kernel_size, strides, conv_normal_initializer, x)

    x = build_MBblock_se_layer(
        intro_filters, squeeze_excite_ratio, x, filters)

    x = build_MBblock_output_layer(
        x, inputs, intro_filters, outro_filters, strides, survival_rate)
    return x


def build_MBblock_input_layer(expand_ratio, filters,
                              conv_normal_initializer, inputs):
    """Builds input layer of the Mobile Inverted Residual Bottleneck
    block.

    # Arguments
        expand_ratio: Int, ratio to expand the conv block in repeats.
        filters: Int, expanded input filters for the blocks to
            construct.
        conv_normal_initializer: Function, that initializes
            convolutional kernels.
        conv_id: Generator, that generates the ID of the convolutional
            layer.
        batch_norm_id: Generator, that generates the ID of the batch
            normalization layer.
        inputs: Tensor, input features for the Mobile Inverted Residual
            Bottleneck block.

    # Returns
        x: Tensor, processed input features for the Mobile Inverted
            Residual Bottleneck block.
    """
    if expand_ratio != 1:
        x = Conv2D(filters, 1, padding='same', use_bias=False,
                   kernel_initializer=conv_normal_initializer)(inputs)
        x = BatchNormalization()(x)
        x = tf.nn.swish(x)
    else:
        x = inputs
    return x


def build_MBblock_depth_wise_conv_layer(kernel_size, strides,
                                        conv_normal_initializer, x):
    """Builds input layer of the Mobile Inverted Residual Bottleneck
    block.

    # Arguments
        kernel_size: Int, size of the kernel filter.
        strides: List, stride of the filter.
        conv_normal_initializer: Function, that initializes
            convolutional kernels.
        batch_norm_id: Generator, that generates the ID of the batch
            normalization layer.
        x: Tensor, input features for the Mobile Inverted Residual
            Bottleneck block.

    # Returns
        x: Tensor, output features from the depthwise convolutional
            layer of the Mobile Inverted Residual Bottleneck block.
    """
    x = DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False,
                        depthwise_initializer=conv_normal_initializer)(x)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    return x


def build_MBblock_se_layer(intro_filters, squeeze_excite_ratio, x, filters):
    """Builds squeeze excitation layer of the Mobile Inverted Residual
    Bottleneck block.

    # Arguments
        intro_filters: Int, input filters for the blocks to construct.
        squeeze_excite_ratio: Float, squeeze excite block ratio.
        x: Tensor, input fro depth wise convolutional layer.
        filters: Int, expanded input filters for the blocks to
            construct.

    # Returns
        x: Tensor, output features from the squeeze excitation layer of
            the Mobile Inverted Residual Bottleneck block.
    """
    num_reduced_filters = max(1, int(intro_filters * squeeze_excite_ratio))
    se = tf.reduce_mean(x, [1, 2], keepdims=True)
    se = Conv2D(num_reduced_filters, 1, padding='same', use_bias=True,
                kernel_initializer=conv_normal_initializer)(se)
    se = tf.nn.swish(se)
    se = Conv2D(filters, 1, padding='same', use_bias=True,
                kernel_initializer=conv_normal_initializer)(se)
    se = tf.sigmoid(se)
    x = se * x
    return x


def build_MBblock_output_layer(x, inputs, intro_filters, outro_filters,
                               strides, survival_rate):
    """Builds output layer of the Mobile Inverted Residual
    Bottleneck block.

    # Arguments
        inputs: Tensor, input features for the Mobile Inverted Residual
            Bottleneck block..
        intro_filters: Int, input filters for the blocks to construct.
        outro_filters: Int, output filters for the blocks to construct.
        conv_id: Generator, that generates the ID of the convolutional
            layer.
        batch_norm_id: Generator, that generates the ID of the batch
            normalization layer.
        x: Tensor, output features from the squeeze excitation layer of
            the Mobile Inverted Residual Bottleneck block.
        strides: List, stride of the filter.
        survival_rate: Float, survival probability to drop

    # Returns
        x: Tensor, output features from the Mobile Inverted Residual
            Bottleneck block.
    """
    x = Conv2D(outro_filters, 1, padding='same', use_bias=False,
               kernel_initializer=conv_normal_initializer)(x)
    x = BatchNormalization()(x)
    if all(s == 1 for s in strides) and intro_filters == outro_filters:
        if survival_rate:
            x = get_drop_connect(x, False, survival_rate)
        x = tf.add(x, inputs)
    return x


def MBconv_block_parameters(block_arg, intro_filters, outro_filters,
                            W_coefficient, D_coefficient, D_divisor, repeats):
    """Compute parameters of the MBConv block.

    # Arguments
        block_arg: Int, specifying the index of the block.
        intro_filters: List, input filters of the blocks.
        outro_filters: List, output filters of the blocks.
        W_coefficient: Float, scaling coefficient for network width.
        D_coefficient: Float, multiplier for the depth of the network.
        D_divisor: Int, multiplier for the depth of the network.
        repeats: Int, number of block repeats.

    # Returns
        intro_filter: Int rounded block input filter.
        outro_filter: Int rounded block output filter.
        repeats: Int rounded repeat of each MBConv block.

    """
    num_intro_filters = intro_filters[block_arg]
    num_outro_filters = outro_filters[block_arg]
    intro_filter = round_filters(num_intro_filters, W_coefficient, D_divisor)
    outro_filter = round_filters(num_outro_filters, W_coefficient, D_divisor)
    repeats = round_repeats(repeats[block_arg], D_coefficient)
    return intro_filter, outro_filter, repeats


def MBconv_block_features(x, block_id, block_arg, survival_rate, kernel_sizes,
                          intro_filter, outro_filter, expand_ratios, strides,
                          repeats, squeeze_excite_ratio):
    """Computes features from a given MBConv block.

    # Arguments
        x: Tensor, output features from the Mobile Inverted Residual
            Bottleneck block.
        block_id: Int, specifying the index of the block.
        survival_rate: Float, survival probability to drop input
            convolution features.
        kernel_sizes: List, kernel size of various
            EfficientNet blocks.
        intro_filters: List, input filters of the blocks.
        outro_filters: List, output filters of the blocks.
        expand_ratio: Int, ratio to expand the conv block in repeats.
        strides: List, strides in height and weight of conv filter.
        repeats: Int, number of block repeats.
        squeeze_excite_ratio: Float, squeeze excite block ratio.
        model_name: String, name of the EfficientNet backbone

    # Returns
        A `Tensor` of type `float32` which is the features from this
            layer.
        block_id: Int, the block identifier.
    """
    stride_value = strides[block_arg]
    for _ in range(repeats):
        x = mobile_inverted_residual_bottleneck_block(
            x, survival_rate, kernel_sizes[block_arg], intro_filter,
            outro_filter, expand_ratios[block_arg], stride_value,
            squeeze_excite_ratio)
        intro_filter, stride_value = outro_filter, [1, 1]
        block_id = block_id + 1
    return x, block_id


def process_feature_maps(
        x, block_arg, intro_filters, outro_filters, W_coefficient,
        D_coefficient, D_divisor, repeats, squeeze_excite_ratio, block_id,
        survival_rate, kernel_sizes, strides, expand_ratios):
    """Computes features from a given MBConv block.

    # Arguments
        x: Tensor, output features from the Mobile Inverted Residual
            Bottleneck block.
        block_arg: Int, specifying the index of the block.
        intro_filters: List, input filters of the blocks.
        outro_filters: List, output filters of the blocks.
        W_coefficient: Float, scaling coefficient for network width.
        D_coefficient: Float, multiplier for the depth of the network.
        D_divisor: Int, multiplier for the depth of the network.
        repeats: Int, number of block repeats.
        squeeze_excite_ratio: Float, squeeze excite block ratio.
        survival_rate: Float, survival probability to drop input
            convolution features.
        kernel_sizes: List, kernel size of various
            EfficientNet blocks.
        strides: List, strides in height and weight of conv filter.
        model_name: String, name of the EfficientNet backbone.
        expand_ratio: Int, ratio to expand the conv block in repeats.

    # Returns
        x: A `Tensor` of type `float32` which is the features from this
            layer.
        block_id: Int, the block identifier.
    """
    parameters = MBconv_block_parameters(
        block_arg, intro_filters, outro_filters, W_coefficient,
        D_coefficient, D_divisor, repeats)
    intro_filter, outro_filter, repeats = parameters

    x, block_id = MBconv_block_features(
        x, block_id, block_arg, survival_rate, kernel_sizes, intro_filter,
        outro_filter, expand_ratios, strides, repeats, squeeze_excite_ratio)

    return x, block_id


def conv_block(image, intro_filters, W_coefficient, D_divisor):
    """Construct the first convolutional layer of EfficientNet.

    # Arguments
        image: Tensor, specifying the input image.
        intro_filters: List, input filters for the blocks to construct.
        W_coefficient: Float, scaling coefficient for network width.
        D_divisor: Int, multiplier for the depth of the network.
        model_name: String, name of the EfficientNet backbone

    # Returns
        x: A `Tensor` of type `float32` which is the features from this
            layer.
    """
    filters = round_filters(intro_filters[0], W_coefficient, D_divisor)
    x = Conv2D(filters, [3, 3], [2, 2], 'same', 'channels_last', [1, 1], 1,
               None, False, conv_normal_initializer)(image)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    return x


def MBconv_blocks(x, kernel_sizes, intro_filters, outro_filters, W_coefficient,
                  D_coefficient, D_divisor, repeats, squeeze_excite_ratio,
                  survival_rate, strides, expand_ratios):
    """Construct the blocks of MBConv: Mobile Inverted Residual
    Bottleneck.

    # Arguments
        x: A `Tensor` of type `float32` which is the features from this
            layer.
        kernel_sizes: List, kernel size of various
            EfficientNet blocks.
        intro_filters: List, input filters of the blocks.
        outro_filters: List, output filters of the blocks.
        W_coefficient: Float, scaling coefficient for network width.
        D_coefficient: Float, multiplier for the depth of the network.
        D_divisor: Int, multiplier for the depth of the network.
        repeats: Int, number of block repeats.
        squeeze_excite_ratio: Float, squeeze excite block ratio.
        survival_rate: Float, survival probability to drop input
            convolution features.
        strides: List, strides in height and weight of conv filter.
        model_name: String, name of the EfficientNet backbone.
        expand_ratio: Int, ratio to expand the conv block in repeats.

    # Returns
        features: A list of Tensor which is the features from this
            layer(s).
    """
    block_id, feature_maps = 0, []
    for block_arg in range(len(kernel_sizes)):
        x, block_id = process_feature_maps(
            x, block_arg, intro_filters, outro_filters, W_coefficient,
            D_coefficient, D_divisor, repeats, squeeze_excite_ratio, block_id,
            survival_rate, kernel_sizes, strides, expand_ratios)

        is_last_block = block_arg == len(kernel_sizes) - 1
        if not is_last_block:
            next_block_stride = strides[block_arg + 1][0]
            if next_block_stride == 2:
                feature_maps.append(x)
        elif is_last_block:
            feature_maps.append(x)
    return feature_maps


def efficientnet(image, model_name, input_shape=(512, 512, 3), D_divisor=8,
                 squeeze_excite_ratio=0.25, kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
                 repeats=[1, 2, 2, 3, 3, 4, 1],
                 intro_filters=[32, 16, 24, 40, 80, 112, 192],
                 outro_filters=[16, 24, 40, 80, 112, 192, 320],
                 expand_ratios=[1, 6, 6, 6, 6, 6, 6],
                 strides=[[1, 1], [2, 2], [2, 2], [2, 2],
                          [1, 1], [2, 2], [1, 1]]):
    """A class implementing tf.keras.Model for EfficientNet. Base paper:
    https://arxiv.org/pdf/1905.11946.pdf
    Initializes an 'Model' instance.
    # Arguments
        image: Tensor, specifying the input image.
        model_name: String, name of the EfficientNet backbone.
        input_shape: Tuple, specifying the shape of the input image.
        D_divisor: Int, multiplier for the depth of the network.
        squeeze_excite_ratio: Float, squeeze excite block ratio.
        kernel_sizes: List, kernel size of the conv block filters.
        repeats: List, number of block repeats.
        intro_filters: List, input filters for the blocks to construct.
        outro_filters: List, output filters for the blocks to construct.
        expand_ratios: List, ratio to expand the conv block in repeats.
        strides: List, strides in height and weight of conv filter.

    # Returns
        features: A list of Tensor which is the features from this
            block.

    # Raises
        ValueError: when repeats is not greater than zero.
    """

    assert (repeats > np.zeros_like(repeats)).sum() == len(repeats)

    scaling_coefficients = get_scaling_coefficients(model_name)
    W_coefficient, D_coefficient, survival_rate = scaling_coefficients

    image = Input(tensor=image, shape=input_shape, name='image')
    x = conv_block(image, intro_filters, W_coefficient, D_divisor)
    x = MBconv_blocks(
        x, kernel_sizes, intro_filters, outro_filters, W_coefficient,
        D_coefficient, D_divisor, repeats, squeeze_excite_ratio,
        survival_rate, strides, expand_ratios)
    return x
