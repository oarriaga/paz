"""Instantiates the EfficientNet architecture using given scaling coefficients.
Reference:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
 https://arxiv.org/abs/1905.11946) (ICML 2019)
"""
import itertools
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Layer, DepthwiseConv2D, Conv2D,
                                     BatchNormalization)
from utils import get_drop_connect


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


class SqueezeExcitation(Layer):
    """Squeeze-and-excitation layer: Recalibrates channel-wise feature responses
    by modelling channel interdependencies as provided in the paper:
    https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, filters, output_filters, name=None):
        """
        # Arguments
            filters: Int, Number of input filters of the Squeeze Excitation
            layer.
            output_filters: Int, Number of output filters of the Squeeze
            Excitation layer.
        """
        super().__init__(name=name)

        self._activation = tf.nn.swish
        # Squeeze and Excitation layer.
        self._reduce = Conv2D(filters, [1, 1], [1, 1], 'same', 'channels_last',
                              (1, 1), 1, None, True, conv_normal_initializer,
                              name='conv2d')
        self._expand = Conv2D(output_filters, [1, 1], [1, 1], 'same',
                              'channels_last', (1, 1), 1, None, True,
                              conv_normal_initializer, name='conv2d_1')

    def call(self, tensor):
        """
        # Arguments
            tensor: Tensor, tensor for forward computation through SE layer.

        # Returns
            output_tensor: Tensor, output tensor after forward computation.
        """
        h_axis, w_axis = 1, 2
        squeeze_excitation_tensor = tf.reduce_mean(
            tensor, [h_axis, w_axis], keepdims=True)
        squeeze_excitation_tensor = self._reduce(squeeze_excitation_tensor)
        squeeze_excitation_tensor = self._activation(squeeze_excitation_tensor)
        squeeze_excitation_tensor = self._expand(squeeze_excitation_tensor)
        return tf.sigmoid(squeeze_excitation_tensor) * tensor


class MobileInvertedResidualBottleNeckBlock(Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck. As provided in
    the paper: https://arxiv.org/pdf/1801.04381.pdf and
    https://arxiv.org/pdf/1905.11946.pdf

    # Attributes
        endpoints: dict. A list of internal tensors.
    """

    def __init__(self, kernel_size, input_filters,
                 output_filters, expand_ratio, strides, squeeze_excite_ratio,
                 name=None):
        """Initializes a MBConv block.

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
        super().__init__(name=name)
        self._activation = tf.nn.swish
        self._kernel_size = kernel_size
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self._strides = strides
        self._squeeze_excite_ratio = squeeze_excite_ratio
        self.endpoints = None
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
        """Builds Squeeze-and-excitation block.

        # Arguments
            filters: Int, output filters of squeeze and excite block.
        """
        num_filter = int(self._input_filters * self._squeeze_excite_ratio)
        num_reduced_filters = max(1, num_filter)
        squeeze_excitation = SqueezeExcitation(
            num_reduced_filters, filters, name='se')
        return squeeze_excitation

    def build_expanded_convolution(self, filters, kernel_size):
        """
        # Arguments
            filters: Int, output filters of expansion conv block.
            kernel: Int, size of the convolution filter.

        # Returns
            expand_conv_block: Tuple of TF layers. Convolution
            layers for expanded convolution block type.
        """
        if self._expand_ratio != 1:
            expand_conv = Conv2D(
                filters, [1, 1], [1, 1], 'same', 'channels_last', (1, 1), 1,
                None, False, conv_normal_initializer,
                name=self.get_conv_name())
        else:
            expand_conv = None
        depthwise_conv = DepthwiseConv2D(
            [kernel_size, kernel_size], self._strides,
            'same', 1, 'channels_last', (1, 1), None, False,
            conv_normal_initializer, name='depthwise_conv2d')
        return expand_conv, depthwise_conv

    def build_conv_layers(self, filters):
        """
        # Arguments
            filters: Int, output filters of fused conv block.

        # Returns
            expand_conv_block: Tuple of TF layers. Convolution
            layers for expanded convolution block type.
        """
        expand_conv_block = self.build_expanded_convolution(
            filters, self._kernel_size)
        return expand_conv_block

    def get_updated_filters(self):
        """Updates filter count depending on the expand ratio.

        # Returns:
            filter: Int, filters count in the successive layers.
        """
        filters = self._input_filters * self._expand_ratio
        return filters

    def build_output_processors(self):
        project_conv = Conv2D(
            self._output_filters, [1, 1], [1, 1], 'same', 'channels_last',
            (1, 1), 1, None, False, conv_normal_initializer,
            name=self.get_conv_name())
        return project_conv

    def _build(self):
        """Builds block according to the arguments."""
        self.conv_id = itertools.count(0)
        self.batch_norm_id = itertools.count(0)
        filters = self.get_updated_filters()
        expand_conv_block = self.build_conv_layers(filters)
        self._expand_conv_layer = expand_conv_block[0]
        self._batch_norm0 = BatchNormalization(
            -1, 0.99, 1e-3, name=self.get_batch_norm_name())
        self._depthwise_conv = expand_conv_block[1]
        self._batch_norm1 = BatchNormalization(
            -1, 0.99, 1e-3, name=self.get_batch_norm_name())
        self._squeeze_excitation = self.build_squeeze_excitation(filters)
        self._project_conv = self.build_output_processors()
        self._batch_norm2 = BatchNormalization(
            -1, 0.99, 1e-3, name=self.get_batch_norm_name())

    def call_conv_layers(self, tensor, training):
        """
        # Arguments
            tensor: Tensor, the input tensor to the body layers of
            MBConv block.
            training: boolean, whether the model is constructed for training.
        """

        if self._expand_ratio != 1:
            tensor = self._expand_conv_layer(tensor)
            tensor = self._batch_norm0(tensor, training=training)
            tensor = self._activation(tensor)
        tensor = self._depthwise_conv(tensor)
        tensor = self._batch_norm1(tensor, training=training)
        tensor = self._activation(tensor)
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
        tensor = self._project_conv(tensor)
        tensor = self._batch_norm2(tensor, training=training)
        tensor = tf.identity(tensor)
        if (all(s == 1 for s in self._strides)
                and self._input_filters == self._output_filters):
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
        x = self.call_conv_layers(x, training)
        if self._squeeze_excitation:
            x = self._squeeze_excitation(x)
        self.endpoints = {'expansion_output': x}
        x = self.call_output_processors(x, training, survival_rate, tensor)
        return x


def round_filters(filters, width_coefficient, depth_divisor, skip=False):
    """Round number of filters based on depth multiplier.

    # Arguments
        filters: Int, filters to be rounded based on depth multiplier.
        with_coefficient: Float, scaling coefficient for network width.
        depth_divisor: Int, multiplier for the depth of the network.
        skip: Bool, skip rounding filters based on multiplier.

    # Returns
        new_filters: Int, rounded filters based on depth multiplier.
    """
    if skip or not width_coefficient:
        return filters
    filters = filters * width_coefficient
    min_depth = depth_divisor
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

    def __init__(self, width_coefficient, depth_divisor, stem_filters,
                 name=None):
        """
        # Arguments
            with_coefficient: Float, scaling coefficient for network width.
            depth_divisor: Int, multiplier for the depth of the network.
            stem_filters: Int, filter count for the stem block.
            name: String, layer name.
        """
        super().__init__(name=name)
        filters = round_filters(
            stem_filters, width_coefficient, depth_divisor)
        self._conv_stem = Conv2D(filters, [3, 3], [2, 2], 'same',
                                 'channels_last', (1, 1), 1, None, False,
                                 conv_normal_initializer)
        self._batch_norm = BatchNormalization(-1, 0.99, 1e-3)
        self._activation = tf.nn.swish

    def call(self, tensor, training):
        """
        # Arguments
            tensor: Tensor, the inputs tensor.
            training: boolean, whether the model is constructed for training.
        """
        output = self._batch_norm(
            self._conv_stem(tensor, training=training), training=training)
        output = self._activation(output)
        return output


class EfficientNet(tf.keras.Model):
    """A class implementing tf.keras.Model for EfficientNet. Base paper:
    https://arxiv.org/pdf/1905.11946.pdf
    """

    def __init__(self, width_coefficient, depth_coefficient,
                 survival_rate, name, num_classes=90, depth_divisor=8,
                 kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
                 num_repeats=[1, 2, 2, 3, 3, 4, 1],
                 input_filters=[32, 16, 24, 40, 80, 112, 192],
                 output_filters=[16, 24, 40, 80, 112, 192, 320],
                 expand_ratios=[1, 6, 6, 6, 6, 6, 6],
                 strides=[[1, 1], [2, 2], [2, 2], [2, 2],
                          [1, 1], [2, 2], [1, 1]],
                 squeeze_excite_ratio=0.25):
        """Initializes an 'Model' instance.

        # Arguments
            blocks_args: Dictionary of BlockArgs to construct block modules.
            num_classes: Int, specifying the number of class in the
            output.
            with_coefficient: Float, scaling coefficient for network width.
            depth_coefficient: Float, scaling coefficient for network depth.
            depth_divisor: Int, multiplier for the depth of the network.
            survival_rate: Float, survival of the final fully connected layer
            units.
            layer.
            kernel_size: Int, kernel size of the conv block filters.
            num_repeats: Int, number of block repeats.
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            expand_ratio: Int, ratio to expand the conv block in repeats.
            strides: List, strides in height and weight of conv filter.
            squeeze_excite_ratio: Float, squeeze excite block ratio.
            num_blocks: Int, number of Mobile bottleneck conv blocks.
            name: A string of layer name.

        # Raises
            ValueError: when blocks_args is not specified as list.
        """
        super().__init__(name=name)

        self._activation = tf.nn.swish
        self._width_coefficient = width_coefficient
        self._depth_coefficient = depth_coefficient
        self._num_classes = num_classes
        self._depth_divisor = depth_divisor
        self._survival_rate = survival_rate
        self._kernel_sizes = kernel_sizes
        self._num_repeats = num_repeats
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratios = expand_ratios
        self._strides = strides
        self._squeeze_excite_ratio = squeeze_excite_ratio
        self._num_blocks = len(kernel_sizes)
        self.endpoints = None
        self._build()

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
            input_filters, self._width_coefficient, self._depth_divisor)
        output_filters = round_filters(
            output_filters, self._width_coefficient, self._depth_divisor)
        return input_filters, output_filters

    def update_block_repeats(self, num_repeats):
        """Update block repeats based on depth multiplier.

        # Arguments
            num_repeats: Int, number of block repeats.
            block_num: Int, Block index.

        # Returns
            num_repeats: Int, number of block repeats.
        """
        num_repeats = round_repeats(num_repeats, self._depth_coefficient)
        return num_repeats

    def add_block_repeats(self, input_filters, output_filters, num_repeat,
                          kernel_size, strides, expand_ratio):
        """
        # Arguments
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            num_repeats: Int, number of block repeats.
            kernel_size: Int, kernel size of the conv block filters.
            strides: List, strides in height and weight of conv filter.
        """
        if num_repeat > 1:
            # rest of blocks with the same block_arg
            input_filters = output_filters
            strides = [1, 1]
        for _ in range(num_repeat - 1):
            self._blocks.append(MobileInvertedResidualBottleNeckBlock(
                kernel_size, input_filters,
                output_filters, expand_ratio, strides,
                self._squeeze_excite_ratio, self.get_block_name()))

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

    def build_blocks(self, input_filters, output_filters, num_repeats,
                     kernel_size, expand_ratio, strides):
        """
        # Arguments
            input_filters: Int, input filters for the blocks to construct.
            output_filters: Int, output filters for the blocks to construct.
            num_repeats: Int, number of block repeats.
            kernel_size: Int, kernel size of the conv block filters.
            expand_ratio: Int, ratio to expand the conv block in repeats.
            strides: List, strides in height and weight of conv filter.
        """

        # The first block needs to take care of stride
        # and filter size increase.
        self._blocks.append(MobileInvertedResidualBottleNeckBlock(
            kernel_size, input_filters,
            output_filters, expand_ratio, strides, self._squeeze_excite_ratio,
            self.get_block_name()))
        self.add_block_repeats(input_filters, output_filters, num_repeats,
                               kernel_size, strides, expand_ratio)

    def _build(self):
        """Builds a model."""
        self._blocks = []

        # Stem part.
        self._stem = Stem(self._width_coefficient, self._depth_divisor,
                          self._input_filters[0])
        self.block_id = itertools.count(0)
        for block_num in range(self._num_blocks):
            assert self._num_repeats[block_num] > 0
            new_filters = self.update_filters(
                self._input_filters[block_num],
                self._output_filters[block_num])
            new_input_filter, new_output_filter = new_filters
            new_repeats = self.update_block_repeats(
                self._num_repeats[block_num])
            self.build_blocks(
                new_input_filter, new_output_filter, new_repeats,
                self._kernel_sizes[block_num], self._expand_ratios[block_num],
                self._strides[block_num])

    def get_survival_rate(self, block_arg):
        """
        # Arguments
            block_arg: Int, block argument of MB conv block.

        # Returns
            survival_rate: Float, survival rate of the MB conv block.
        """
        survival_rate = self._survival_rate
        if survival_rate:
            drop_rate = 1 - survival_rate
            drop_rate_changer = float(block_arg) / len(self._blocks)
            survival_rate = 1 - drop_rate * drop_rate_changer
        return survival_rate

    def reduce_block(self, block_arg, tensor, reduction_arg):
        """
        # Arguments
            block_arg: Int, block argument of MB conv block.
            tensor: Tensor, outputs from MB conv block sequences.
            reduction_arg: Int, reduction from the previous block outputs.

        # Returns
            is_reduction: Bool, flag to further add reduction tensors.
            reduction_arg: Int, reduction from the previous block outputs.
        """
        is_reduction = False
        if ((block_arg == len(self._blocks) - 1) or
                self._blocks[block_arg + 1]._strides[0] > 1):
            is_reduction = True
            reduction_arg = reduction_arg + 1
        return is_reduction, reduction_arg

    def call_stem(self, tensor, training):
        """
        # Arguments
           tensor: input tensors.
           training: boolean, whether the model is constructed for training.

        # Returns
           tensors: Tensor, output from the efficientnet stem.
        """
        tensor = self._stem(tensor, training)
        self.endpoints['stem'] = tensor
        return tensor

    def call_blocks(self, tensor, training):
        """
        # Arguments
           tensor: input tensors.
           training: boolean, whether the model is constructed for training.

        # Returns
           tensors: Tensor, output from the efficientnet blocks.
        """
        reduction_arg = 0
        for block_arg, block in enumerate(self._blocks):
            survival_rate = self.get_survival_rate(block_arg)
            tensor = block(tensor, training, survival_rate)
            self.endpoints['block_%s' % block_arg] = tensor
            is_reduction, reduction_arg = self.reduce_block(
                block_arg, tensor, reduction_arg)
            if is_reduction:
                self.endpoints['reduction_%s' % reduction_arg] = tensor
        self.endpoints['features'] = tensor
        return tensor

    def collate_feature_levels(self, tensor):
        """
        # Arguments:
            tensor: Tensor, efficientnet output from blocks.

        # Returns:
            tensor: List, contains tensors representing different level
            features.
        """
        tensor = [tensor]
        for block_arg in range(5):
            key_name = 'reduction_' + str(block_arg + 1)
            tensor = tensor + [self.endpoints.get(key_name)]
        return tensor

    def call(self, tensor, training):
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
        tensor = self.call_stem(tensor, training)
        tensor = self.call_blocks(tensor, training)
        tensor = self.collate_feature_levels(tensor)
        return tensor
