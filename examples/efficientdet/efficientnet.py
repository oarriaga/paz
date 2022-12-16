import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, DepthwiseConv2D


def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters using depth divisor.

    # Arguments
        filters: Int, filters to be rounded.
        width_coefficient: Float, width coefficient.
        depth_divisor: Int, network depth divisor.

    # Returns
        new_filters: Int, rounded filters.
    """
    filters = filters * width_coefficient
    half_depth = depth_divisor / 2
    threshold = (int(filters + half_depth) // depth_divisor) * depth_divisor
    new_filters = int(max(depth_divisor, threshold))
    if new_filters < 0.9 * filters:
        new_filters = int(new_filters + depth_divisor)
    return new_filters


def round_repeats(repeats, depth_coefficient):
    """Round number of block repeats using depth divisor.

    # Arguments
        repeats: Int, number of multiplier blocks.
        depth_coefficient: Float, network depth scaling coefficient.

    # Returns
        Int: Rounded block repeats.
    """
    return int(math.ceil(depth_coefficient * repeats))


def normal_kernel_initializer(shape, dtype=None):
    """Initialize convolutional kernel using zero
    centred Gaussian distribution.

    # Arguments
        shape: variable shape.
        dtype: variable dtype.

    # Returns
        variable initialization.
    """
    kernel_height, kernel_width, _, outro_filters = shape
    fan_output = int(kernel_height * kernel_width * outro_filters)
    return tf.random.normal(shape, 0.0, np.sqrt(2.0 / fan_output), dtype)


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


def MB_block(inputs, intro_filters, outro_filters, strides, kernel_size,
             survival_rate, expand_ratio, SE_ratio):
    """Initialize Mobile Inverted Residual Bottleneck block.

    # Arguments
        inputs: Tensor, input features to MB block.
        intro_filters: Int, block's input filters.
        outro_filters: Int, block's output filters.
        strides: List, conv block filter strides.
        kernel_size: Int, conv block kernel size.
        survival_rate: Float, survival probability to drop features.
        expand_ratio: Int, conv block expansion ratio.
        SE_ratio: Float, squeeze excite block ratio.

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

    # MB Block Input ----------------------------------------------------------
    if expand_ratio != 1:
        x = Conv2D(filters, 1, padding='same', use_bias=False,
                   kernel_initializer=normal_kernel_initializer)(inputs)
        x = BatchNormalization()(x)
        x = tf.nn.swish(x)
    else:
        x = inputs

    # MB Block Convolution  ---------------------------------------------------
    x = DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False,
                        depthwise_initializer=normal_kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)

    # MB Block Squeeze Excitation ---------------------------------------------
    num_reduced_filters = max(1, int(intro_filters * SE_ratio))
    SE = tf.reduce_mean(x, [1, 2], keepdims=True)
    SE = Conv2D(num_reduced_filters, 1, padding='same', use_bias=True,
                kernel_initializer=normal_kernel_initializer)(SE)
    SE = tf.nn.swish(SE)
    SE = Conv2D(filters, 1, padding='same', use_bias=True,
                kernel_initializer=normal_kernel_initializer)(SE)
    SE = tf.sigmoid(SE)
    x = SE * x

    # MB Block Output ---------------------------------------------------------
    x = Conv2D(outro_filters, 1, padding='same', use_bias=False,
               kernel_initializer=normal_kernel_initializer)(x)
    x = BatchNormalization()(x)
    all_strides_one = all(stride == 1 for stride in strides)
    if all_strides_one and intro_filters == outro_filters:
        if survival_rate:
            x = apply_drop_connect(x, False, survival_rate)
        x = tf.add(x, inputs)
    return x


def MBconv_block_features(x, survival_rate, SE_ratio, intro_filter,
                          outro_filter, repeats, kernel_size, expand_ratio,
                          stride):
    """Computes given MBConv block's features.

    # Arguments
        x: Tensor, input features.
        survival_rate: Float, survival probability to drop features.
        SE_ratio: Float, block's squeeze excite ratio.
        intro_filter: Int, block's input filter.
        outro_filter: Int, block's output filter.
        repeats: Int, number of block repeats.
        kernel_size: Int, kernel size.
        expand_ratio: Int, MBConv block expansion ratio.
        stride: Int, filter strides.

    # Returns
        Tensor: Output features.
        block_id: Int, block identifier.
    """
    args = (kernel_size, survival_rate, expand_ratio, SE_ratio)
    x = MB_block(x, intro_filter, outro_filter, stride, *args)
    for _ in range(1, repeats):
        x = MB_block(x, outro_filter, outro_filter, [1, 1], *args)
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
    filters = round_filters(intro_filters[0], width_coefficient, depth_divisor)
    x = Conv2D(filters, [3, 3], [2, 2], 'same', 'channels_last', [1, 1], 1,
               None, False, normal_kernel_initializer)(image)
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
        D_divisor: Int, network depth divisor.
        repeats: Int, number of block repeats.
        SE_ratio: Float, block's squeeze excite ratio.
        survival_rate: Float, survival probability to drop features.
        strides: List, filter strides.
        expand_ratios: Int, MBConv block expansion ratio.

    # Returns
        feature_maps: List, of output features.
    """
    iterator_1 = zip(intro_filters, outro_filters, repeats)
    iterator_2 = zip(kernel_sizes, expand_ratios, strides)
    feature_append_mask = [stride[0] == 2 for stride in strides[1:]]
    feature_append_mask.append(True)
    feature_maps = []

    for args_1, args_2, should_append_feature in zip(iterator_1, iterator_2,
                                                     feature_append_mask):
        intro_filter, outro_filter, repeat = args_1
        intro_filter = round_filters(intro_filter, W_coefficient, D_divisor)
        outro_filter = round_filters(outro_filter, W_coefficient, D_divisor)
        repeat = round_repeats(repeat, D_coefficient)
        args_1 = intro_filter, outro_filter, repeat
        x = MBconv_block_features(x, survival_rate, SE_ratio, *args_1, *args_2)
        if should_append_feature:
            feature_maps.append(x)

    return feature_maps


def EFFICIENTNET(image, scaling_coefficients, D_divisor=8, SE_ratio=0.25,
                 kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
                 repeats=[1, 2, 2, 3, 3, 4, 1],
                 intro_filters=[32, 16, 24, 40, 80, 112, 192],
                 outro_filters=[16, 24, 40, 80, 112, 192, 320],
                 expand_ratios=[1, 6, 6, 6, 6, 6, 6],
                 strides=[[1, 1], [2, 2], [2, 2], [2, 2],
                          [1, 1], [2, 2], [1, 1]]):
    """A class implementing EfficientNet.

    # Arguments
        image: Tensor of shape `(batch_size, input_shape)`, input image.
        scaling_coefficients: List, EfficientNet scaling coefficients.
        input_shape: Tuple, input image shape.
        D_divisor: Int, network depth divisor.
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

    W_coefficient, D_coefficient, survival_rate = scaling_coefficients
    x = conv_block(image, intro_filters, W_coefficient, D_divisor)
    x = MBconv_blocks(
        x, kernel_sizes, intro_filters, outro_filters,
        W_coefficient, D_coefficient, D_divisor, repeats,
        SE_ratio, survival_rate, strides, expand_ratios)
    return x
