import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     DepthwiseConv2D, Input)


def get_scaling_coefficients(model_name):
    """Model name to scaling coefficients.

    # Arguments
        model_name: Str, EfficientNet backbone name.

    # Returns
        List: with width coefficient, depth coefficient, survival rate
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
    """Round number of filters using depth multiplier.

    # Arguments
        filters: Int, filters to be rounded.
        W_coefficient: Float, width coefficient.
        D_divisor: Int, network depth multiplier.

    # Returns
        new_filters: Int, rounded filters.
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
    """Round number of block repeats using depth multiplier.

    # Arguments
        repeats: Int, number of multiplier blocks.
        D_coefficient: Float, network depth scaling coefficient.

    # Returns
        new_repeats: Int, rounded block repeats.
    """
    new_repeats = int(math.ceil(D_coefficient * repeats))
    return new_repeats


def kernel_initializer(shape, dtype=None):
    """Initialize convolutional kernels.

    # Arguments
        shape: variable shape.
        dtype: variable dtype.

    # Returns
        variable initialization.
    """
    kernel_H, kernel_W, _, outro_filters = shape
    fan_output = int(kernel_H * kernel_W * outro_filters)
    return tf.random.normal(shape, 0.0, np.sqrt(2.0 / fan_output), dtype)


def get_drop_connect(x, is_training, survival_rate):
    """Drops conv with given survival probability.

    # Arguments
        x: Tensor, input feature map to undergo drop connection.
        is_training: Bool specifying training phase.
        survival_rate: Float, survival probability to drop features.

    # Returns
        output: Tensor, output feature map after drop connect.

    # References
        [Deep Networks with Stochastic Depth]
        (https://arxiv.org/pdf/1603.09382.pdf)
    """
    if not is_training:
        return x
    batch_size = tf.shape(x)[0]
    kwargs = {"shape": [batch_size, 1, 1, 1], "dtype": x.dtype}
    random_tensor = survival_rate + tf.random.uniform(**kwargs)
    binary_tensor = tf.floor(random_tensor)
    output = (x / survival_rate) * binary_tensor
    return output


def MB_block(inputs, survival_rate, kernel_size, intro_filters,
             outro_filters, expand_ratio, strides, SE_ratio):
    """Initialize Mobile Inverted Residual Bottleneck block.

    # Arguments
        kernel_size: Int, conv block kernel size.
        repeats: Int, number of block repeats.
        intro_filters: Int, block's input filters.
        outro_filters: Int, block's output filters.
        expand_ratio: Int, conv block expansion ratio.
        strides: List, conv block filter strides.
        SE_ratio: Float, squeeze excite block ratio.
        num_blocks: Int, number of Mobile bottleneck conv blocks.

    # Returns
        x: Tensor, output features.

    # References
        [MobileNetV2: Inverted Residuals and Linear Bottlenecks]
        (https://arxiv.org/pdf/1801.04381.pdf)
        [EfficientNet: Rethinking Model Scaling for
         Convolutional Neural Networks]
        (https://arxiv.org/pdf/1905.11946.pdf)
    """
    filters = intro_filters * expand_ratio
    x = input_MB_block(inputs, expand_ratio, filters)
    x = conv_MB_block(x, kernel_size, strides)
    x = MB_SE_block(x, intro_filters, SE_ratio, filters)
    args = (x, inputs, intro_filters, outro_filters, strides, survival_rate)
    x = output_MB_block(*args)
    return x


def input_MB_block(inputs, expand_ratio, filters):
    """Builds Mobile Inverted Residual Bottleneck block's input layer.

    # Arguments
        inputs: Tensor, input features.
        expand_ratio: Int, conv block expansion ratio.
        filters: Int, block's expanded input filters.

    # Returns
        x: Tensor, output features.
    """
    if expand_ratio != 1:
        x = Conv2D(filters, 1, padding='same', use_bias=False,
                   kernel_initializer=kernel_initializer)(inputs)
        x = BatchNormalization()(x)
        x = tf.nn.swish(x)
    else:
        x = inputs
    return x


def conv_MB_block(x, kernel_size, strides):
    """Builds Mobile Inverted Residual Bottleneck block's conv layer.

    # Arguments
        x: Tensor, input features.
        kernel_size: Int, kernel size.
        strides: List, filter strides.

    # Returns
        x: Tensor, output features.
    """
    x = DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False,
                        depthwise_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    return x


def MB_SE_block(x, intro_filters, SE_ratio, filters):
    """Builds Mobile Inverted Residual Bottleneck block's
    Squeeze Excitation layer.

    # Arguments
        x: Tensor, input features.
        intro_filters: Int, block's input filters.
        SE_ratio: Float, block's squeeze excite ratio.
        filters: Int, block's expanded input filters.

    # Returns
        x: Tensor, output features.
    """
    num_reduced_filters = max(1, int(intro_filters * SE_ratio))
    SE = tf.reduce_mean(x, [1, 2], keepdims=True)
    SE = Conv2D(num_reduced_filters, 1, padding='same', use_bias=True,
                kernel_initializer=kernel_initializer)(SE)
    SE = tf.nn.swish(SE)
    SE = Conv2D(filters, 1, padding='same', use_bias=True,
                kernel_initializer=kernel_initializer)(SE)
    SE = tf.sigmoid(SE)
    x = SE * x
    return x


def output_MB_block(x, inputs, intro_filters, outro_filters,
                    strides, survival_rate):
    """Builds Mobile Inverted Residual Bottleneck block's output layer.

    # Arguments
        x: Tensor, input features from squeeze excitation layer.
        inputs: Tensor, input features from Mobile Inverted Residual
            Bottleneck block.
        intro_filters: Int, block's input filters.
        outro_filters: Int, block's output filters.
        strides: List, filter strides.
        survival_rate: Float, survival probability to drop features.

    # Returns
        x: Tensor, output features.
    """
    x = Conv2D(outro_filters, 1, padding='same', use_bias=False,
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    if all(s == 1 for s in strides) and intro_filters == outro_filters:
        if survival_rate:
            x = get_drop_connect(x, False, survival_rate)
        x = tf.add(x, inputs)
    return x


def compute_MBconv_block_parameters(block_arg, intro_filters, outro_filters,
                                    W_coefficient, D_coefficient, D_divisor,
                                    repeats):
    """Compute MBConv block parameters.

    # Arguments
        block_arg: Int, block index.
        intro_filters: Int, block's input filters.
        outro_filters: Int, block's output filters.
        W_coefficient: Float, width coefficient.
        D_coefficient: Float, network depth scaling coefficient.
        D_divisor: Int, network depth multiplier.
        repeats: Int, number of block repeats.

    # Returns
        intro_filter: Int, rounded block's input filter.
        outro_filter: Int, rounded block's output filter.
        repeats: Int, rounded repeats of each MBConv block.

    """
    num_intro_filters = intro_filters[block_arg]
    num_outro_filters = outro_filters[block_arg]
    intro_filter = round_filters(num_intro_filters, W_coefficient, D_divisor)
    outro_filter = round_filters(num_outro_filters, W_coefficient, D_divisor)
    repeat = round_repeats(repeats[block_arg], D_coefficient)
    return intro_filter, outro_filter, repeat


def MBconv_block_features(x, block_id, block_arg, survival_rate, kernel_sizes,
                          intro_filter, outro_filter, expand_ratios, strides,
                          repeats, SE_ratio):
    """Computes given MBConv block's features.

    # Arguments
        x: Tensor, input features.
        block_id: Int, MBConv block index.
        block_arg: Int, block index.
        survival_rate: Float, survival probability to drop features.
        kernel_sizes: List, kernel sizes.
        intro_filters: Int, block's input filters.
        outro_filters: Int, block's output filters.
        expand_ratios: Int, MBConv block expansion ratio.
        strides: List, filter strides.
        repeats: Int, number of block repeats.
        SE_ratio: Float, block's squeeze excite ratio.

    # Returns
        Tensor: Output features.
        block_id: Int, block identifier.
    """
    kernel_size = kernel_sizes[block_arg]
    expand_ratio = expand_ratios[block_arg]
    stride = strides[block_arg]

    for _ in range(repeats):
        x = MB_block(x, survival_rate, kernel_size, intro_filter,
                     outro_filter, expand_ratio, stride, SE_ratio)
        intro_filter, stride = outro_filter, [1, 1]

    block_id += repeats
    return x, block_id


def process_feature_maps(x, block_arg, intro_filter, outro_filter, repeat,
                         SE_ratio, block_id, survival_rate, kernel_sizes,
                         strides, expand_ratios):
    """Process given MBConv block's features.

    # Arguments
        x: Tensor, input features.
        block_arg: Int, block index.
        intro_filters: Int, block's input filters.
        outro_filters: Int, block's output filters.
        repeat: Int, number of block repeats.
        SE_ratio: Float, block's squeeze excite ratio.
        block_id: Int, MBConv block index.
        survival_rate: Float, survival probability to drop features.
        kernel_sizes: List, kernel sizes.
        strides: List, filter strides.
        expand_ratios: Int, MBConv block expansion ratio.

    # Returns
        x: Tensor, output features.
        block_id: Int, MBConv block identifier.
    """
    args = (x, block_id, block_arg, survival_rate, kernel_sizes, intro_filter,
            outro_filter, expand_ratios, strides, repeat, SE_ratio)
    x, block_id = MBconv_block_features(*args)

    return x, block_id


def conv_block(image, intro_filters, W_coefficient, D_divisor):
    """Builds EfficientNet's first convolutional layer.

    # Arguments
        image: Tensor, the input image.
        intro_filters: Int, block's input filters.
        W_coefficient: Float, width coefficient.
        D_divisor: Int, network depth multiplier.

    # Returns
        x: Tensor, output features.
    """
    filters = round_filters(intro_filters[0], W_coefficient, D_divisor)
    x = Conv2D(filters, [3, 3], [2, 2], 'same', 'channels_last', [1, 1], 1,
               None, False, kernel_initializer)(image)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    return x


def MBconv_blocks(x, kernel_sizes, intro_filters, outro_filters, W_coefficient,
                  D_coefficient, D_divisor, repeats, SE_ratio,
                  survival_rate, strides, expand_ratios):
    """Builds MBConv blocks.

    # Arguments
        x: Tensor, input features.
        kernel_sizes: List, kernel sizes.
        intro_filters: Int, block's input filters.
        outro_filters: Int, block's output filters.
        W_coefficient: Float, width coefficient.
        D_coefficient: Float, network depth scaling coefficient.
        D_divisor: Int, network depth multiplier.
        repeats: Int, number of block repeats.
        SE_ratio: Float, block's squeeze excite ratio.
        survival_rate: Float, survival probability to drop features.
        strides: List, filter strides.
        expand_ratios: Int, MBConv block expansion ratio.

    # Returns
        feature_maps: List, of output features.
    """
    block_id, feature_maps = 0, []
    for block_arg in range(len(kernel_sizes)):
        args = (block_arg, intro_filters, outro_filters, W_coefficient,
                D_coefficient, D_divisor, repeats)
        parameters = compute_MBconv_block_parameters(*args)
        intro_filter, outro_filter, repeat = parameters

        args = (x, block_arg, intro_filter, outro_filter, repeat, SE_ratio,
                block_id, survival_rate, kernel_sizes, strides, expand_ratios)
        x, block_id = process_feature_maps(*args)

        is_last_block = block_arg == len(kernel_sizes) - 1
        if not is_last_block:
            next_block_stride = strides[block_arg + 1][0]
            if next_block_stride == 2:
                feature_maps.append(x)
        elif is_last_block:
            feature_maps.append(x)
    return feature_maps


def efficientnet(image, model_name, input_shape=(512, 512, 3), D_divisor=8,
                 SE_ratio=0.25, kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
                 repeats=[1, 2, 2, 3, 3, 4, 1],
                 intro_filters=[32, 16, 24, 40, 80, 112, 192],
                 outro_filters=[16, 24, 40, 80, 112, 192, 320],
                 expand_ratios=[1, 6, 6, 6, 6, 6, 6],
                 strides=[[1, 1], [2, 2], [2, 2], [2, 2],
                          [1, 1], [2, 2], [1, 1]]):
    """A class implementing EfficientNet.

    # Arguments
        image: Tensor, the input image.
        model_name: Str, EfficientNet backbone name.
        input_shape: Tuple, input image shape.
        D_divisor: Int, network depth multiplier.
        SE_ratio: Float, block's squeeze excite ratio.
        kernel_sizes: List, kernel sizes.
        repeats: Int, number of block repeats.
        intro_filters: Int, block's input filters.
        outro_filters: Int, block's output filters.
        expand_ratios: Int, MBConv block expansion ratio.
        strides: List, filter strides.

    # Returns
        x: List, output features.

    # Raises
        ValueError: when repeats is not greater than zero.

    # References
        [EfficientNet: Rethinking Model Scaling for
         Convolutional Neural Networks]
        (https://arxiv.org/pdf/1905.11946.pdf)
    """
    assert (repeats > np.zeros_like(repeats)).sum() == len(repeats)

    scaling_coefficients = get_scaling_coefficients(model_name)
    W_coefficient, D_coefficient, survival_rate = scaling_coefficients

    image = Input(tensor=image, shape=input_shape, name='image')
    x = conv_block(image, intro_filters, W_coefficient, D_divisor)
    args = (x, kernel_sizes, intro_filters, outro_filters, W_coefficient,
            D_coefficient, D_divisor, repeats, SE_ratio,
            survival_rate, strides, expand_ratios)
    x = MBconv_blocks(*args)
    return x
