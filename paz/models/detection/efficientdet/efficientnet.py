import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, DepthwiseConv2D


def EFFICIENTNET(image, scaling_coefficients, D_divisor=8, excite_ratio=0.25,
                 kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
                 repeats=[1, 2, 2, 3, 3, 4, 1],
                 intro_filters=[32, 16, 24, 40, 80, 112, 192],
                 outro_filters=[16, 24, 40, 80, 112, 192, 320],
                 expand_ratios=[1, 6, 6, 6, 6, 6, 6],
                 strides=[[1, 1], [2, 2], [2, 2], [2, 2],
                          [1, 1], [2, 2], [1, 1]]):
    """A function implementing EfficientNet.

    # Arguments
        image: Tensor of shape `(batch_size, input_shape)`, input image.
        scaling_coefficients: List, EfficientNet scaling coefficients.
        D_divisor: Int, network depth divisor.
        excite_ratio: Float, block's squeeze excite ratio.
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

    W_coefficient, D_coefficient, survival_rate = scaling_coefficients
    x = conv_block(image, intro_filters, W_coefficient, D_divisor)
    x = MBconv_blocks(
        x, kernel_sizes, intro_filters, outro_filters,
        W_coefficient, D_coefficient, D_divisor, repeats,
        excite_ratio, survival_rate, strides, expand_ratios)
    return x


def conv_block(image, intro_filters, width_coefficient, depth_divisor):
    """Builds EfficientNet's first convolutional layer.

    # Arguments
        image: Tensor of shape `(batch_size, input_shape)`, input image.
        intro_filters: Int, block's input filters.
        width_coefficient: Float, width coefficient.
        depth_divisor: Int, network depth divisor.

    # Returns
        x: Tensor, output features.
    """
    filters = scale_filters(intro_filters[0], width_coefficient, depth_divisor)
    x = Conv2D(filters, [3, 3], [2, 2], 'same', 'channels_last', [1, 1], 1,
               None, False, kernel_initializer)(image)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    return x


def scale_filters(filters, width_coefficient, depth_divisor):
    """Scales filters using depth divisor.

    # Arguments
        filters: Int, filters to be rounded.
        width_coefficient: Float, width coefficient.
        depth_divisor: Int, network depth divisor.

    # Returns
        scaled_filters: Int, scaled filters.
    """
    filters_scaled_by_width = filters * width_coefficient
    half_depth = depth_divisor / 2
    filters_rounded = int(filters_scaled_by_width + half_depth)
    filters_standardized = filters_rounded // depth_divisor
    threshold = filters_standardized * depth_divisor
    scaled_filters = int(max(depth_divisor, threshold))
    if scaled_filters < 0.9 * filters_scaled_by_width:
        scaled_filters = int(scaled_filters + depth_divisor)
    return scaled_filters


def kernel_initializer(shape, dtype=None):
    """Initialize convolutional kernel with
    zero centred Gaussian distribution.

    # Arguments
        shape: variable shape.
        dtype: variable dtype.

    # Returns
        variable initialization.
    """
    kernel_height, kernel_width, _, outro_filters = shape
    fan_output = int(kernel_height * kernel_width * outro_filters)
    return tf.random.normal(shape, 0.0, np.sqrt(2.0 / fan_output), dtype)


def MBconv_blocks(x, kernel_sizes, intro_filters, outro_filters, W_coefficient,
                  D_coefficient, D_divisor, repeats, excite_ratio,
                  survival_rate, strides, expand_ratios):
    """Builds EfficientNet's MBConv blocks.
    MBConv stands for Mobile Inverted Bottleneck Convolution.

    # Arguments
        x: Tensor, input features.
        kernel_sizes: List, kernel sizes.
        intro_filters: Int, block's input filters.
        outro_filters: Int, block's output filters.
        W_coefficient: Float, width coefficient.
        D_coefficient: Float, network depth scaling coefficient.
        D_divisor: Int, network depth divisor.
        repeats: Int, number of block repeats.
        excite_ratio: Float, block's squeeze excite ratio.
        survival_rate: Float, survival probability to drop features.
        strides: List, filter strides.
        expand_ratios: List, MBConv block's expansion ratio.

    # Returns
        feature_maps: List, of output features.
    """
    feature_append_mask = [stride[0] == 2 for stride in strides[1:]]
    feature_append_mask.append(True)

    intro_filters = [scale_filters(intro_filter, W_coefficient, D_divisor)
                     for intro_filter in intro_filters]
    outro_filters = [scale_filters(outro_filter, W_coefficient, D_divisor)
                     for outro_filter in outro_filters]
    repeats = [round_repeats(repeat, D_coefficient) for repeat in repeats]
    excite_ratios = [excite_ratio] * len(outro_filters)
    survival_rates = [survival_rate] * len(outro_filters)

    iterator_1 = list(zip(intro_filters, outro_filters, strides, repeats))
    iterator_2 = list(zip(kernel_sizes, survival_rates, expand_ratios,
                          excite_ratios))
    feature_maps = []
    for feature_arg, args in enumerate(zip(iterator_1, iterator_2)):
        repeat_args, block_args = args
        x = MB_repeat(x, *repeat_args, block_args)
        if feature_append_mask[feature_arg]:
            feature_maps.append(x)
    return feature_maps


def round_repeats(repeats, depth_coefficient):
    """Round number of block repeats using depth divisor.

    # Arguments
        repeats: Int, number of multiplier blocks.
        depth_coefficient: Float, network depth scaling coefficient.

    # Returns
        Int: Rounded block repeats.
    """
    return int(math.ceil(depth_coefficient * repeats))


def MB_repeat(x, intro_filter, outro_filter, stride, repeats, block_args):
    """Computes given MBConv block's features.

    # Arguments
        x: Tensor, input features.
        intro_filter: Int, block's input filter.
        outro_filter: Int, block's output filter.
        stride: Int, filter strides.
        repeats: Int, number of block repeats.
        block_args: Tuple, holding kernel_sizes, survival_rates,
            expand_ratios, excite_ratios.

    # Returns
        Tensor: Output features.
    """
    for _ in range(repeats):
        x = MB_block(x, intro_filter, outro_filter, stride, *block_args)
        intro_filter, stride = outro_filter, [1, 1]
    return x


def MB_block(inputs, intro_filters, outro_filters, strides, kernel_size,
             survival_rate, expand_ratio, excite_ratio):
    """Initialize Mobile Inverted Residual Bottleneck block.

    # Arguments
        inputs: Tensor, input features to MB block.
        intro_filters: Int, block's input filters.
        outro_filters: Int, block's output filters.
        strides: List, conv block filter strides.
        kernel_size: Int, conv block kernel size.
        survival_rate: Float, survival probability to drop features.
        expand_ratio: Int, conv block expansion ratio.
        excite_ratio: Float, squeeze excite block ratio.

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
    x = MB_input(inputs, filters, expand_ratio)
    x = MB_convolution(x, kernel_size, strides)
    x = MB_squeeze_excitation(x, intro_filters, expand_ratio, excite_ratio)
    x = MB_output(x, inputs, intro_filters, outro_filters, strides,
                  survival_rate)
    return x


def MB_input(inputs, filters, expand_ratio):
    if expand_ratio != 1:
        x = MB_conv2D(inputs, filters, use_bias=False)
        x = BatchNormalization()(x)
        x = tf.nn.swish(x)
    else:
        x = inputs
    return x


def MB_conv2D(x, filters, use_bias=False):
    kwargs = {'padding': 'same', 'kernel_initializer': kernel_initializer}
    return Conv2D(filters, 1, use_bias=use_bias, **kwargs)(x)


def MB_convolution(x, kernel_size, strides):
    kwargs = {'padding': 'same', 'depthwise_initializer': kernel_initializer}
    x = DepthwiseConv2D(kernel_size, strides, use_bias=False, **kwargs)(x)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    return x


def MB_squeeze_excitation(x, intro_filters, expand_ratio, excite_ratio):
    num_reduced_filters = max(1, int(intro_filters * excite_ratio))
    SE = tf.reduce_mean(x, [1, 2], keepdims=True)
    SE = MB_conv2D(SE, num_reduced_filters, use_bias=True)
    SE = tf.nn.swish(SE)
    SE = MB_conv2D(SE, intro_filters * expand_ratio, use_bias=True)
    SE = tf.sigmoid(SE)
    return SE * x


def MB_output(x, inputs, intro_filters, outro_filters, strides, survival_rate):
    x = MB_conv2D(x, outro_filters, use_bias=False)
    x = BatchNormalization()(x)
    all_strides_one = all(stride == 1 for stride in strides)
    if all_strides_one and intro_filters == outro_filters:
        if survival_rate:
            x = apply_drop_connect(x, False, survival_rate)
        x = tf.add(x, inputs)
    return x


def apply_drop_connect(x, is_training, survival_rate):
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
        output = x
    else:
        batch_size = tf.shape(x)[0]
        kwargs = {"shape": [batch_size, 1, 1, 1], "dtype": x.dtype}
        random_tensor = survival_rate + tf.random.uniform(**kwargs)
        binary_tensor = tf.floor(random_tensor)
        output = (x * binary_tensor) / survival_rate
    return output
