"""Instantiates the EfficientNet architecture using given scaling coefficients.
Reference:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
 https://arxiv.org/abs/1905.11946) (ICML 2019)
"""
import itertools
import math
import numpy as np
import six
import tensorflow as tf
from tensorflow.keras.layers import Layer, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from utils import get_activation, get_drop_connect


def conv_kernel_initializer(shape, dtype=None):
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


def dense_kernel_initializer(shape, dtype=None):
    """Initialization for dense kernels.
    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_output',
                                      distribution='uniform').
    It is written out explicitly here for clarity.

    # Arguments
        shape: shape of variable
        dtype: dtype of variable

    # Returns
        an initialization for the variable
    """
    span = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(shape, -span, span, dtype=dtype)


def superpixel_kernel_initializer(shape, dtype='float32'):
    """Initializes superpixel kernels.
    This is inspired by space-to-depth transformation that
    is mathematically equivalent before and after the
    transformation. But we do the space-to-depth via a
    convolution. Moreover, we make the layer trainable
    instead of direct transform, we can initialization it
    this way so that the model can learn not to do anything
    but keep it mathematically equivalent,
    when improving performance.

    # Arguments
        shape: shape of variable
        dtype: dtype of variable

    # Returns
        an initialization for the variable
    """
    #  use input depth to make superpixel kernel.
    depth = shape[-2]
    span_x = np.arange(2)
    span_y = np.arange(2)
    span_z = np.arange(depth)
    mesh = np.array(np.meshgrid(span_x, span_y, span_z)).T.reshape(-1, 3).T
    filters = np.zeros([2, 2, depth, 4 * depth], dtype=dtype)
    filters[mesh[0], mesh[1], mesh[2], 4 * mesh[2] + 2 * mesh[0] + mesh[1]] = 1
    return filters


class Squeeze_Excitation(Layer):
    """Squeeze-and-excitation layer."""
    def __init__(self, local_pooling, activation, filters,
                 output_filters, name=None):
        """
        # Arguments
            local_pooling: bool, flag to use local pooling of features.
            activation: String, activation function name.
            filters: Int, Number of input filters of the Squeeze Excitation
            layer.
            output_filters: Int, Number of output filters of the Squeeze
            Excitation layer.
        """
        super().__init__(name=name)

        self._local_pooling = local_pooling
        self._activation = activation

        # Squeeze and Excitation layer.
        self._reduce = Conv2D(filters, [1, 1], [1, 1], 'same',
                              'channels_last', (1, 1), 1, None, True,
                              conv_kernel_initializer, name='conv2d')
        self._expand = Conv2D(output_filters, [1, 1], [1, 1], 'same',
                              'channels_last', (1, 1), 1, None, True,
                              conv_kernel_initializer, name='conv2d_1')

    def call(self, tensor):
        """
        # Arguments
            tensor: Tensor, tensor for forward computation through SE layer.

        # Returns
            output_tensor: Tensor, output tensor after forward computation.
        """
        h_axis, w_axis = [1, 2]
        if self._local_pooling:
            squeeze_excitation_tensor = tf.nn.avg_pool(
                tensor, [1, tensor.shape[h_axis], tensor.shape[w_axis], 1],
                [1, 1, 1, 1], 'VALID')
        else:
            squeeze_excitation_tensor = tf.reduce_mean(
                tensor, [h_axis, w_axis], keepdims=True)
        squeeze_excitation_tensor = self._expand(get_activation(
            self._reduce(squeeze_excitation_tensor), self._activation))
        return tf.sigmoid(squeeze_excitation_tensor) * tensor


class SuperPixel(Layer):
    """Super pixel layer."""

    def __init__(self, input_filters, batch_norm, activation, name=None):
        """
        # Arguments
            input_filters: Int, input filters for the blocks to construct.
            batch_norm: TF BatchNormalization layer.
            activation: String, activation function name.
            name: layer name.
        """

        super().__init__(name=name)
        self._superpixel = Conv2D(input_filters, [2, 2], [2, 2],
                                  'same', 'channels_last', (1, 1), 1, None,
                                  False, superpixel_kernel_initializer,
                                  name='conv2d')
        self._batch_norm_superpixel = batch_norm(-1, 0.99, 1e-3)
        self._activation = activation

    def call(self, tensor, training):
        """
        # Arguments
            tensor: Tensor, tensor for forward computation through SE layer.
            training: Bool, indicating whether forward computation is happening

        # Returns
            output_tensor: Tensor, output tensor after forward computation.
        """
        output = self._superpixel(tensor)
        output = self._batch_norm_superpixel(output, training)
        output = get_activation(output, self._activation)
        return output


class MBConvBlock(Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.

    # Attributes
        endpoints: dict. A list of internal tensors.
    """

    def __init__(self, local_pooling, batch_norm, activation,
                 use_squeeze_excitation, clip_projection_output,
                 kernel_size, input_filters, output_filters, expand_ratio,
                 strides, squeeze_excite_ratio, fused_conv, super_pixel,
                 use_skip_connection, name=None):
        """Initializes a MBConv block.

        # Arguments
            local_pooling: bool, flag to use local pooling of features.
            batch_norm: TF BatchNormalization layer.
            activation: String, activation function name.
            use_squeeze_excitation: bool, flag to use squeeze and excitation
            layer.
            clip_projection_output: String, flag to clip output.
            kernel_size: Int, kernel size of the conv block filters.
            num_repeats: Int, number of block repeats.
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            expand_ratio: Int, ratio to expand the conv block in repeats.
            strides: List, strides in height and weight of conv filter.
            squeeze_excite_ratio: Float, squeeze excite block ratio.
            fused_conv: Int, flag to select fused conv or expand conv.
            super_pixel: Int, superpixel value.
            use_skip_connection: bool, flag to skip connect.
            num_blocks: Int, number of Mobile bottleneck conv blocks.
            name: layer name.
        """
        super().__init__(name=name)
        self._local_pooling = local_pooling
        self._batch_norm = batch_norm
        self._activation = activation
        self._clip_projection_output = clip_projection_output
        self._kernel_size = kernel_size
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self._strides = strides
        self._squeeze_excite_ratio = squeeze_excite_ratio
        self._fused_conv = fused_conv
        self._super_pixel = super_pixel
        self._use_skip_connection = use_skip_connection
        self._has_squeeze_excitation = (
                use_squeeze_excitation
                and self._squeeze_excite_ratio is not None
                and 0 < self._squeeze_excite_ratio <= 1)
        self.endpoints = None
        # Builds the block according to arguments.
        self._build()

    @property
    def block_args(self):
        return self._block_args

    def get_conv_name(self):
        if not next(self.conv_id):
            name_appender = ""
        else:
            name_appender = '_' + str(next(self.conv_id) // 2)
        name = 'conv2d' + name_appender
        return name

    def get_batch_norm_name(self):
        if not next(self.batch_norm_id):
            name_appender = ""
        else:
            name_appender = '_' + str(next(self.batch_norm_id) // 2)
        name = 'batch_normalization' + name_appender
        return name

    def build_squeeze_excitation(self, filters):
        """
        Builds Squeeze-and-excitation block.

        # Arguments
            filters: Int, output filters of squeeze and excite block.
        """
        batch_norm = self._batch_norm(
            -1, 0.99, 1e-3, name=self.get_batch_norm_name())
        if self._has_squeeze_excitation:
            num_filter = int(self._input_filters * self._squeeze_excite_ratio)
            num_reduced_filters = max(1, num_filter)
            squeeze_excitation = Squeeze_Excitation(
                self._local_pooling, self._activation,
                num_reduced_filters, filters, name='se')
        else:
            squeeze_excitation = None
        return batch_norm, squeeze_excitation

    def build_super_pixel(self):
        if self._super_pixel == 1:
            super_pixel = SuperPixel(
                self._input_filters, self._batch_norm,
                self._activation, name='super_pixel')
        else:
            super_pixel = None
        return super_pixel

    def build_fused_convolution(self, filters, kernel_size):
        """
        # Arguments
            filters: Int, output filters of fused conv block.
            kernel: Int, size of the convolution filter.

        # Returns
            fused_conv_block: TF Layers. Convolution layers for
            fused convolution block type.
        """
        fused_conv = Conv2D(
            filters, [kernel_size, kernel_size],
            self._strides, 'same', 'channels_last',
            (1, 1), 1, None, False, conv_kernel_initializer,
            name=self.get_conv_name())
        return fused_conv

    def build_expanded_convolution(self, filters, kernel_size):
        """
        # Arguments
            filters: Int, output filters of expansion conv block.
            kernel: Int, size of the convolution filter.

        # Returns
            expand_conv_block: Tuple of TF layers. Convolution
            layers for expanded convolution block type.
        """
        expand_conv = None
        batch_norm0 = None
        if self._expand_ratio != 1:
            expand_conv = Conv2D(
                filters, [1, 1], [1, 1], 'same', 'channels_last',
                (1, 1), 1, None, False, conv_kernel_initializer,
                name=self.get_conv_name())
            batch_norm0 = self._batch_norm(
                -1, 0.99, 1e-3, name=self.get_batch_norm_name())
        depthwise_conv = DepthwiseConv2D(
            [kernel_size, kernel_size], self._strides,
            'same', 1, 'channels_last', (1, 1), None, False,
            conv_kernel_initializer, name='depthwise_conv2d')
        return expand_conv, batch_norm0, depthwise_conv

    def build_conv_layers(self, filters):
        """
        # Arguments
            filters: Int, output filters of fused conv block.

        # Returns
            fused_conv_block: TF Layers. Convolution layers for
            fused convolution block type.
            expand_conv_block: Tuple of TF layers. Convolution
            layers for expanded convolution block type.
        """
        fused_conv_block = None
        expand_conv_block = (None, None, None)
        kernel_size = self._kernel_size
        if self._fused_conv:
            fused_conv_block = self.build_fused_convolution(
                filters, kernel_size)
        else:
            expand_conv_block = self.build_expanded_convolution(
                filters, kernel_size)
        return fused_conv_block, expand_conv_block

    def get_updated_filters(self):
        """Updates filter count depending on the expand ratio.
        # Returns:
        filter: Int, filters count in the successive layers.
        """
        block_input_filters = self._input_filters
        block_expand_ratio = self._expand_ratio
        filters = block_input_filters * block_expand_ratio
        return filters

    def build_output_processors(self):
        filters = self._output_filters
        project_conv = Conv2D(
            filters, [1, 1], [1, 1], 'same', 'channels_last', (1, 1), 1, None,
            False, conv_kernel_initializer, name=self.get_conv_name())
        batch_norm = self._batch_norm(
            -1, 0.99, 1e-3, name=self.get_batch_norm_name())
        return project_conv, batch_norm

    def _build(self):
        """Builds block according to the arguments."""
        self.conv_id = itertools.count(0)
        self.batch_norm_id = itertools.count(0)
        self.super_pixel_layer = self.build_super_pixel()
        filters = self.get_updated_filters()
        fused_conv_block, expand_conv_block = self.build_conv_layers(filters)
        self._fused_conv_layer = fused_conv_block
        self._expand_conv_layer = expand_conv_block[0]
        self._batch_norm0 = expand_conv_block[1]
        self._depthwise_conv = expand_conv_block[2]
        squeeze_block = self.build_squeeze_excitation(filters)
        self._batch_norm1 = squeeze_block[0]
        self._squeeze_excitation = squeeze_block[1]
        output_block = self.build_output_processors()
        self._project_conv = output_block[0]
        self._batch_norm2 = output_block[1]

    def call_conv_layers(self, tensor, training):
        """
        # Arguments
            tensor: Tensor, the input tensor to the body layers of
            MBConv block.
            training: boolean, whether the model is constructed for training.
        """
        if self._fused_conv:
            tensor = self._batch_norm1(
                self._fused_conv_layer(tensor), training=training)
            tensor = get_activation(tensor, self._activation)
        else:
            if self._expand_ratio != 1:
                tensor = self._batch_norm0(
                    self._expand_conv_layer(tensor), training=training)
                tensor = get_activation(tensor, self._activation)

            tensor = self._batch_norm1(
                self._depthwise_conv(tensor), training=training)
            tensor = get_activation(tensor, self._activation)
        return tensor

    def call_output_processors(self, tensor, training,
                               survival_rate, initial_tensor):
        """
        # Arguments
            tensor: Tensor, the input tensor to be output processed.
            training: boolean, whether the model is constructed for training.
            survival_rate: float, between 0 to 1, drop connect rate.
            initial_tensor: Tensor, the input tensor to the MBConvBlock.

        # Returns
            A output tensor.
        """
        tensor = self._batch_norm2(
            self._project_conv(tensor), training=training)
        tensor = tf.identity(tensor)
        if self._clip_projection_output:
            tensor = tf.clip_by_value(tensor, -6, 6)
        if self._use_skip_connection:
            if (all(s == 1 for s in self._strides)
                    and self._input_filters
                    == self._output_filters):
                if survival_rate:
                    tensor = get_drop_connect(tensor, training, survival_rate)
                tensor = tf.add(tensor, initial_tensor)
        return tensor

    def call(self, tensor, training, survival_rate):
        """Implementation of call().
        # Arguments
            tensor: Tensor, the inputs tensor.
            training: boolean, whether the model is constructed for training.
            survival_rate: float, between 0 to 1, drop connect rate.

        # Returns
            A output tensor.
        """
        x = tensor
        if self.super_pixel_layer:
            x = self.super_pixel_layer(x, training)
        x = self.call_conv_layers(x, training)
        if self._squeeze_excitation:
            x = self._squeeze_excitation(x)
        self.endpoints = {'expansion_output': x}
        x = self.call_output_processors(x, training, survival_rate, tensor)
        return x


class MBConvBlockWithoutDepthwise(MBConvBlock):
    pass


def round_filters(filters, width_coefficient, depth_divisor,
                  min_depth, skip=False):
    """Round number of filters based on depth multiplier.
    # Arguments
        filters: Int, filters to be rounded based on depth multiplier.
        with_coefficient: Float, scaling coefficient for network width.
        depth_divisor: Int, multiplier for the depth of the network.
        min_depth: Int, minimum depth of the network.
        skip: Bool, skip rounding filters based on multiplier.

    # Returns
        new_filters: Int, rounded filters based on depth multiplier.
    """
    if skip or not width_coefficient:
        return filters
    filters = filters * width_coefficient
    min_depth = min_depth or depth_divisor
    half_depth = depth_divisor / 2
    threshold = int(filters + half_depth) // depth_divisor * depth_divisor
    new_filters = max(min_depth, threshold)
    if new_filters < 0.9 * filters:
        new_filters = new_filters + depth_divisor
    new_filters = int(new_filters)
    return new_filters


def round_repeats(repeats, depth_coefficient, skip=False):
    """Round number of filters based on depth multiplier.

    # Arguments
        repeats: Int, number of repeats of multiplier blocks.
        depth_coefficient: Float, scaling coefficient for network depth.
        skip: Bool, skip rounding filters based on multiplier.

    # Returns
        new_repeats: Int, repeats of blocks based on multiplier.
    """
    if skip or not depth_coefficient:
        return repeats
    new_repeats = int(math.ceil(depth_coefficient * repeats))
    return new_repeats


class Stem(Layer):
    """Stem layer at the beginning of the network."""

    def __init__(self, width_coefficient, depth_divisor, min_depth,
                 fix_head_stem, batch_norm, activation, stem_filters,
                 name=None):
        """
        # Arguments
            with_coefficient: Float, scaling coefficient for network width.
            depth_divisor: Int, multiplier for the depth of the network.
            min_depth: Int, minimum depth of the network.
            fix_head_stem: bool, flag to fix head and stem branches.
            batch_norm: TF BatchNormalization layer.
            activation: String, activation function name.
            stem_filters: Int, filter count for the stem block.
            name: String, layer name.
        """
        super().__init__(name=name)
        filters = round_filters(
            stem_filters, width_coefficient, depth_divisor, min_depth,
            fix_head_stem)
        self._conv_stem = Conv2D(filters, [3, 3], [2, 2], 'same',
                                 'channels_last', (1, 1), 1, None, False,
                                 conv_kernel_initializer)
        self._batch_norm = batch_norm(-1, 0.99, 1e-3)
        self._activation = activation

    def call(self, tensor, training):
        """
        # Arguments
            tensor: Tensor, the inputs tensor.
            training: boolean, whether the model is constructed for training.
        """
        output = self._batch_norm(
            self._conv_stem(tensor, training=training), training=training)
        output = get_activation(output, self._activation)
        return output


class Head(Layer):
    """Head layer for network outputs."""

    def __init__(self, width_coefficient, depth_divisor, min_depth,
                 fix_head_stem, batch_norm, activation, num_classes,
                 dropout_rate, local_pooling, name=None):
        """
        # Arguments
            with_coefficient: Float, scaling coefficient for network width.
            depth_divisor: Int, multiplier for the depth of the network.
            min_depth: Int, minimum depth of the network.
            fix_head_stem: bool, flag to fix head and stem branches.
            batch_norm: TF BatchNormalization layer.
            activation: String, activation function name.
            num_classes: Int, specifying the number of class in the
            output.
            dropout_rate: Float, dropout rate for final fully connected layers.
            local_pooling: bool, flag to use local pooling of features.
            name: String, layer name.
        """
        super().__init__(name=name)

        self.endpoints = {}
        conv_filters = round_filters(
            1280, width_coefficient, depth_divisor, min_depth, fix_head_stem)
        self._conv_head = Conv2D(
            conv_filters, [1, 1], [1, 1], 'same', 'channels_last', (1, 1), 1,
            None, False, conv_kernel_initializer, name='conv2d')
        self._batch_norm = batch_norm(-1, 0.99, 1e-3)
        self._activation = activation
        self._avg_pooling = GlobalAveragePooling2D(data_format='channels_last')
        if num_classes:
            self._fully_connected = Dense(
                num_classes, kernel_initializer=dense_kernel_initializer)
        else:
            self._fully_connected = None
        if dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(dropout_rate)
        else:
            self._dropout = None
        self.h_axis, self.w_axis = ([1, 2])
        self._local_pooling = local_pooling

    def call_local_pooling_head(self, tensor, training, pool_features):
        """
        # Arguments
            tensor: Tensor, the inputs tensor.
            training: boolean, whether the model is constructed for training.
            pool_features: Bool, flag to decide feature pooling from tensor.

        # Returns
            tensor: Tensor, local pooled head output tensor.
        """
        shape = tensor.get_shape().as_list()
        kernel_size = [1, shape[self.h_axis], shape[self.w_axis], 1]
        tensor = tf.nn.avg_pool(
            tensor, kernel_size, [1, 1, 1, 1], 'VALID')
        self.endpoints['pooled_features'] = tensor
        if not pool_features:
            if self._dropout:
                tensor = self._dropout(tensor, training=training)
            self.endpoints['global_pool'] = tensor
            if self._fully_connected:
                tensor = tf.squeeze(tensor, [self.h_axis, self.w_axis])
                tensor = self._fully_connected(tensor)
            self.endpoints['head'] = tensor
        return tensor

    def call_avg_pooling_head(self, tensor, training, pool_features):
        """
        # Arguments
            tensor: Tensor, the inputs tensor.
            training: boolean, whether the model is constructed for training.
            pool_features: Bool, flag to decide feature pooling from tensor.

        # Returns
            tensor: Tensor, average pooled head output tensor.
        """
        tensor = self._avg_pooling(tensor)
        self.endpoints['pooled_features'] = tensor
        if not pool_features:
            if self._dropout:
                tensor = self._dropout(tensor, training=training)
            self.endpoints['global_pool'] = tensor
            if self._fully_connected:
                tensor = self._fully_connected(tensor)
            self.endpoints['head'] = tensor
        return tensor

    def call(self, tensor, training, pool_features):
        """Call the head layer.
        # Arguments
            tensor: Tensor, the inputs tensor.
            training: boolean, whether the model is constructed for training.
            pool_features: Bool, flag to decide feature pooling from tensor.
        # Returns
            tensor: Tensor, Pooled head output tensor.
        """
        outputs = self._batch_norm(self._conv_head(tensor), training=training)
        outputs = get_activation(outputs, self._activation)
        self.endpoints['head_1x1'] = outputs
        if self._local_pooling:
            outputs = self.call_local_pooling_head(
                outputs, training, pool_features)
        else:
            outputs = self.call_avg_pooling_head(
                outputs, training, pool_features)
        return outputs


class EfficientNet(tf.keras.Model):
    """A class implementing tf.keras.Model for EfficientNet."""

    def __init__(self, dropout_rate, width_coefficient, depth_coefficient,
                 survival_rate, name, data_format='channels_last',
                 num_classes=90, depth_divisor=8, min_depth=None,
                 use_squeeze_excitation=True, local_pooling=None,
                 clip_projection_output=False, fix_head_stem=None,
                 kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
                 num_repeats=[1, 2, 2, 3, 3, 4, 1],
                 input_filters=[32, 16, 24, 40, 80, 112, 192],
                 output_filters=[16, 24, 40, 80, 112, 192, 320],
                 expand_ratios=[1, 6, 6, 6, 6, 6, 6],
                 strides=[[1, 1], [2, 2], [2, 2], [2, 2],
                          [1, 1], [2, 2], [1, 1]],
                 squeeze_excite_ratio=0.25,
                 use_skip_connection=True,
                 conv_type=0, fused_conv=0, super_pixel=0, num_blocks=7,
                 activation='swish',
                 batch_norm=BatchNormalization):
        """Initializes an 'Model' instance.

        # Arguments
            blocks_args: Dictionary of BlockArgs to construct block modules.
            dropout_rate: Float, dropout rate for final fully connected layers.
            data_format: String, Data format 'channels_first' or
            'channels_last'.
            num_classes: Int, specifying the number of class in the
            output.
            with_coefficient: Float, scaling coefficient for network width.
            depth_coefficient: Float, scaling coefficient for network depth.
            depth_divisor: Int, multiplier for the depth of the network.
            min_depth: Int, minimum depth of the network.
            survival_rate: Float, survival of the final fully connected layer
            units.
            activation: String, activation function name.
            batch_norm: TF BatchNormalization layer.
            use_squeeze_excitation: bool, flag to use squeeze and excitation
            layer.
            local_pooling: bool, flag to use local pooling of features.
            clip_projection_output: String, flag to clip output.
            fix_head_stem: bool, flag to fix head and stem branches.
            kernel_size: Int, kernel size of the conv block filters.
            num_repeats: Int, number of block repeats.
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            expand_ratio: Int, ratio to expand the conv block in repeats.
            strides: List, strides in height and weight of conv filter.
            squeeze_excite_ratio: Float, squeeze excite block ratio.
            use_skip_connection: bool, flag to skip connect.
            conv_type: Int, flag to select convolution type.
            fused_conv: Int, flag to select fused conv or expand conv.
            super_pixel: Int, superpixel value.
            num_blocks: Int, number of Mobile bottleneck conv blocks.
            name: A string of layer name.

        # Raises
            ValueError: when blocks_args is not specified as list.
        """
        super().__init__(name=name)

        self._activation = activation
        self._batch_norm = batch_norm
        self._fix_head_stem = fix_head_stem
        self._width_coefficient = width_coefficient
        self._depth_coefficient = depth_coefficient
        self._dropout_rate = dropout_rate
        self._data_format = data_format
        self._num_classes = num_classes
        self._depth_divisor = depth_divisor
        self._min_depth = min_depth
        self._survival_rate = survival_rate
        self._use_squeeze_excitation = use_squeeze_excitation
        self._local_pooling = local_pooling
        self._clip_projection_output = clip_projection_output
        self._fix_head_stem = fix_head_stem
        self._kernel_sizes = kernel_sizes
        self._num_repeats = num_repeats
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratios = expand_ratios
        self._strides = strides
        self._squeeze_excite_ratio = squeeze_excite_ratio
        self._use_skip_connection = use_skip_connection
        self._conv_type = conv_type
        self._fused_conv = fused_conv
        self._super_pixel = super_pixel
        self._num_blocks = num_blocks
        self.endpoints = None
        self._build()

    def _get_conv_block(self, conv_type):
        """
        # Arguments
            conv_type: Int, key deciding the Convolution block type
            0 - Mobile Bottleneck block with depthwise convolution.
            1 - Mobile Bottleneck block without depthwise convolution.

        # Returns
            conv_block_map: Convolution block.
        """
        conv_block_map = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}
        return conv_block_map[conv_type]

    def get_block_name(self):
        name = 'blocks_%d' % next(self.block_id)
        return name

    def update_filters(self, input_filters, output_filters):
        """Update block input and output filters based on depth multiplier.
        # Arguments
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.

        # Returns
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
        """
        input_filters = round_filters(
            input_filters, self._width_coefficient,
            self._depth_divisor, self._min_depth)
        output_filters = round_filters(
            output_filters, self._width_coefficient,
            self._depth_divisor, self._min_depth)
        return input_filters, output_filters

    def update_block_repeats(self, num_repeats, block_num):
        """Update block repeats based on depth multiplier.
        # Arguments
            num_repeats: Int, number of block repeats.
            block_num: Int, Block index.

        # Returns
            num_repeats: Int, number of block repeats.
        """
        blocks_repeat_limit = self._num_blocks - 1
        block_flag = (block_num == 0 or block_num == blocks_repeat_limit)
        if self._fix_head_stem and block_flag:
            num_repeats = num_repeats
        else:
            num_repeats = round_repeats(num_repeats, self._depth_coefficient)
        return num_repeats

    def add_block_repeats(self, input_filters, output_filters, num_repeat,
                          kernel_size, strides, expand_ratio, super_pixel):
        """
        # Arguments
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            num_repeats: Int, number of block repeats.
            kernel_size: Int, kernel size of the conv block filters.
            strides: List, strides in height and weight of conv filter.
            expand_ratio: Int, ratio to expand the conv block in repeats.
            super_pixel: Int, superpixel value.
        """
        conv_block = self._get_conv_block(self._conv_type)
        if num_repeat > 1:
            # rest of blocks with the same block_arg
            input_filters = output_filters
            strides = [1, 1]
        for _ in range(num_repeat - 1):
            self._blocks.append(conv_block(
                self._local_pooling, self._batch_norm, self._activation,
                self._use_squeeze_excitation, self._clip_projection_output,
                kernel_size, input_filters, output_filters,
                expand_ratio, strides, self._squeeze_excite_ratio,
                self._fused_conv, super_pixel, self._use_skip_connection,
                self.get_block_name()))

    def update_block_depth(self, input_filters, output_filters,
                           kernel_size, strides):
        """
        # Arguments
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            kernel_size: Int, kernel size of the conv block filters.
            strides: List, strides in height and weight of conv filter.

        # Returns
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            kernel_size: Int, kernel size of the conv block filters.
        """
        depth_factor = int(4 / strides[0] / strides[1])
        if depth_factor > 1:
            kernel_size = (kernel_size + 1) // 2
        else:
            kernel_size = kernel_size
        input_filters = input_filters * depth_factor
        output_filters = output_filters * depth_factor
        return input_filters, output_filters, kernel_size

    def add_stride2_block(self, input_filters, output_filters, kernel_size,
                          expand_ratio, super_pixel):
        """
        # Arguments
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            kernel_size: Int, kernel size of the conv block filters.
            expand_ratio: Int, ratio to expand the conv block in repeats.
            super_pixel: Int, superpixel value.

        # Returns
            super_pixel: Int, superpixel value.
        """
        conv_block = self._get_conv_block(self._conv_type)
        strides = [1, 1]
        self._blocks.append(conv_block(
            self._local_pooling, self._batch_norm,self._activation,
            self._use_squeeze_excitation, self._clip_projection_output,
            kernel_size, input_filters, output_filters, expand_ratio,
            strides, self._squeeze_excite_ratio, self._fused_conv,
            super_pixel, self._use_skip_connection,
            name=self.get_block_name()))
        super_pixel = 0
        return super_pixel

    def build_super_pixel_blocks(self, input_filters, output_filters,
                                 kernel_size, strides, expand_ratio,
                                 super_pixel):
        """
        # Arguments
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            num_repeats: Int, number of block repeats.
            kernel_size: Int, kernel size of the conv block filters.
            strides: List, strides in height and weight of conv filter.
            expand_ratio: Int, ratio to expand the conv block in repeats.
            super_pixel: Int, superpixel value.

        # Returns
            kernel_size: Int, kernel size of the conv block filters.
            super_pixel: Int, superpixel value.
        """
        conv_block = self._get_conv_block(self._conv_type)
        # if superpixel, adjust filters, kernels, and strides.
        updated_parameters = self.update_block_depth(
            input_filters, output_filters, kernel_size, strides)
        new_input_filters, new_output_filters, kernel_size = updated_parameters
        # if the first block has stride-2 and superpixel transformation
        if strides[0] == 2 and strides[1] == 2:
            updated_super_pixel = self.add_stride2_block(
                new_input_filters, new_output_filters, kernel_size,
                expand_ratio, super_pixel)
            super_pixel = updated_super_pixel
        elif super_pixel == 1:
            self._blocks.append(conv_block(
                self._local_pooling, self._batch_norm, self._activation,
                self._use_squeeze_excitation, self._clip_projection_output,
                kernel_size, new_input_filters, new_output_filters,
                expand_ratio, strides, self._squeeze_excite_ratio,
                self._fused_conv, super_pixel, self._use_skip_connection,
                self.get_block_name()))
            super_pixel = 2
        else:
            self._blocks.append(conv_block(
                self._local_pooling, self._batch_norm, self._activation,
                self._use_squeeze_excitation, self._clip_projection_output,
                kernel_size, new_input_filters, new_output_filters,
                expand_ratio, strides, self._squeeze_excite_ratio,
                self._fused_conv, super_pixel,
                self._use_skip_connection, self.get_block_name()))

        return kernel_size, super_pixel

    def build_blocks(self, input_filters, output_filters, num_repeats,
                     kernel_size, expand_ratio, strides, super_pixel):
        """
        # Arguments
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            num_repeats: Int, number of block repeats.
            kernel_size: Int, kernel size of the conv block filters.
            expand_ratio: Int, ratio to expand the conv block in repeats.
            strides: List, strides in height and weight of conv filter.
            super_pixel: Int, superpixel value.
        """

        # The first block needs to take care of stride
        # and filter size increase.
        conv_block = self._get_conv_block(self._conv_type)
        if not self._super_pixel:  # no super_pixel at all
            self._blocks.append(conv_block(
                self._local_pooling, self._batch_norm, self._activation,
                self._use_squeeze_excitation, self._clip_projection_output,
                kernel_size, input_filters, output_filters, expand_ratio,
                strides, self._squeeze_excite_ratio, self._fused_conv,
                super_pixel, self._use_skip_connection,
                self.get_block_name()))
        else:
            new_params = self.build_super_pixel_blocks(
                input_filters, output_filters, kernel_size, strides,
                expand_ratio, super_pixel)
            kernel_size, super_pixel = new_params
        self.add_block_repeats(input_filters, output_filters, num_repeats,
                               kernel_size, strides, expand_ratio, super_pixel)

    def _build(self):
        """Builds a model."""
        self._blocks = []

        # Stem part.
        self._stem = Stem(
            self._width_coefficient, self._depth_divisor, self._min_depth,
            self._fix_head_stem, self._batch_norm, self._activation,
            self._input_filters[0])
        self.block_id = itertools.count(0)
        for block_num in range(self._num_blocks):
            assert self._num_repeats[block_num] > 0
            assert self._super_pixel in [0, 1, 2]
            new_filters = self.update_filters(
                self._input_filters[block_num],
                self._output_filters[block_num])
            new_input_filter, new_output_filter = new_filters
            new_repeats = self.update_block_repeats(
                self._num_repeats[block_num], block_num)
            self.build_blocks(
                new_input_filter, new_output_filter, new_repeats,
                self._kernel_sizes[block_num], self._expand_ratios[block_num],
                self._strides[block_num], self._super_pixel)
        # Head part.
        self._head = Head(
            self._width_coefficient, self._depth_divisor, self._min_depth,
            self._fix_head_stem, self._batch_norm, self._activation,
            self._num_classes, self._dropout_rate, self._local_pooling)

    def call(self, tensor, training, return_base=None, pool_features=False):
        """Implementation of call().

        # Arguments
            tensor: input tensors.
            training: boolean, whether the model is constructed for training.
            return_base: build the base feature network only.
            pool_features: build the base network for
            features extraction (after 1x1 conv layer and global
            pooling, but before dropout and fully connected head).

        # Returns
            output tensors: Tensor, output from the efficientnet backbone.
        """
        self.endpoints = {}
        reduction_arg = 0

        # Calls Stem layers
        outputs = self._stem(tensor, training)
        self.endpoints['stem'] = outputs

        # Call blocks
        for block_arg, block in enumerate(self._blocks):
            is_reduction = False
            if block._super_pixel == 1 and block_arg == 0:
                reduction_arg = reduction_arg + 1
                self.endpoints['reduction_%s' % reduction_arg] = outputs

            elif ((block_arg == len(self._blocks) - 1) or
                    self._blocks[block_arg + 1]._strides[0] > 1):
                is_reduction = True
                reduction_arg = reduction_arg + 1

            survival_rate = self._survival_rate
            if survival_rate:
                drop_rate = 1 - survival_rate
                drop_rate_changer = float(block_arg) / len(self._blocks)
                survival_rate = 1 - drop_rate * drop_rate_changer
            outputs = block(
                outputs, training=training, survival_rate=survival_rate)
            self.endpoints['block_%s' % block_arg] = outputs
            if is_reduction:
                self.endpoints['reduction_%s' % reduction_arg] = outputs
            if block.endpoints:
                for block_key, block_value in six.iteritems(block.endpoints):
                    self.endpoints[
                        'block_%s/%s' % (block_arg, block_key)] = block_value
                    if is_reduction:
                        self.endpoints[
                            'reduction_%s/%s' % (reduction_arg, block_key)
                        ] = block_value
        self.endpoints['features'] = outputs

        if not return_base:
            outputs = self._head(outputs, training, pool_features)
            self.endpoints.update(self._head.endpoints)

        return [outputs] + list(
            filter(
                lambda endpoint: endpoint is not None,
                [
                    self.endpoints.get('reduction_1'),
                    self.endpoints.get('reduction_2'),
                    self.endpoints.get('reduction_3'),
                    self.endpoints.get('reduction_4'),
                    self.endpoints.get('reduction_5'),
                ],
            )
        )
