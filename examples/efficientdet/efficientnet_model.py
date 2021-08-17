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
from utils import get_activation, get_drop_connect


GlobalParams_arg = ['dropout_rate', 'data_format', 'num_classes',
                    'width_coefficient', 'depth_coefficient',
                    'depth_divisor', 'min_depth', 'survival_rate',
                    'activation', 'batch_norm', 'use_se', 'local_pooling',
                    'condconv_num_experts', 'clip_projection_output',
                    'blocks_args', 'fix_head_stem']
GlobalParams = dict.fromkeys(GlobalParams_arg, None)

BlockArgs_arg = ['kernel_size', 'num_repeat', 'input_filters',
                 'output_filters', 'expand_ratio', 'id_skip',
                 'strides', 'se_ratio', 'conv_type', 'fused_conv',
                 'super_pixel', 'condconv']
BlockArgs = dict.fromkeys(BlockArgs_arg, None)


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.initializers.variance_scaling uses a truncated normal with
    a corrected standard deviation.

    # Arguments
        shape: shape of variable
        dtype: dtype of variable
        partition_info: unused

    # Returns
        an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, output_filters = shape
    fan_output = int(kernel_height * kernel_width * output_filters)
    return tf.random.normal(shape, 0.0, np.sqrt(2.0 / fan_output), dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.
    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_output',
                                      distribution='uniform').
    It is written out explicitly here for clarity.

    # Arguments
        shape: shape of variable
        dtype: dtype of variable
        partition_info: unused

    # Returns
        an initialization for the variable
    """
    del partition_info
    span = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(shape, -span, span, dtype=dtype)


def superpixel_kernel_initializer(shape, dtype='float32', partition_info=None):
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
        partition_info: unused

    # Returns
        an initialization for the variable
    """
    del partition_info
    #  use input depth to make superpixel kernel.
    depth = shape[-2]
    filters = np.zeros([2, 2, depth, 4 * depth], dtype=dtype)
    span_x = np.arange(2)
    span_y = np.arange(2)
    span_z = np.arange(depth)
    mesh = np.array(np.meshgrid(span_x, span_y, span_z)).T.reshape(-1, 3).T
    filters[mesh[0], mesh[1], mesh[2], 4 * mesh[2] + 2 * mesh[0] + mesh[1]] = 1
    return filters


class SE(Layer):
    """Squeeze-and-excitation layer."""
    def __init__(self, global_params, se_filters, output_filters, name=None):
        """
        # Arguments
            global_params: GlobalParams, a set of global parameters.
            se_filters: Int, Number of input filters of the SE layer.
            output_filters: Int, Number of output filters of the SE layer.
        """
        super().__init__(name=name)

        self._local_pooling = global_params["local_pooling"]
        self._activation = global_params["activation"]

        # Squeeze and Excitation layer.
        self._se_reduce = Conv2D(se_filters, [1, 1], [1, 1], 'same',
                                 'channels_last', (1, 1), 1, None, True,
                                 conv_kernel_initializer, name='conv2d')
        self._se_expand = Conv2D(output_filters, [1, 1], [1, 1], 'same',
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
            se_tensor = tf.nn.avg_pool(
                tensor, [1, tensor.shape[h_axis], tensor.shape[w_axis], 1],
                [1, 1, 1, 1], 'VALID')
        else:
            se_tensor = tf.reduce_mean(tensor, [h_axis, w_axis], keepdims=True)
        se_tensor = self._se_expand(get_activation(
            self._se_reduce(se_tensor), self._activation))
        print('Built SE %s : %s', self.name, se_tensor.shape)
        return tf.sigmoid(se_tensor) * tensor


class SuperPixel(Layer):
    """Super pixel layer."""

    def __init__(self, block_args, global_params, name=None):
        """
        # Arguments
            block_args: Dictionary of BlockArgs to construct block modules.
            global_params: GlobalParams, a set of global parameters.
            name: layer name.
        """

        super().__init__(name=name)
        self._superpixel = Conv2D(block_args["input_filters"], [2, 2], [2, 2],
                                  'same', 'channels_last', (1, 1), 1, None,
                                  False, superpixel_kernel_initializer,
                                  name='conv2d')
        self._batch_norm_superpixel = global_params["batch_norm"](
            -1, 0.99, 1e-3)
        self._activation = global_params["activation"]

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

    def __init__(self, block_args, global_params, name=None):
        """Initializes a MBConv block.

        # Arguments
            block_args: Dictionary of BlockArgs to construct block modules.
            global_params: GlobalParams, a set of global parameters.
            name: layer name.
        """
        super().__init__(name=name)
        self._block_args = block_args
        self._global_params = global_params
        self._local_pooling = global_params["local_pooling"]
        self._batch_norm = global_params["batch_norm"]
        self._condconv_num_experts = global_params["condconv_num_experts"]
        self._activation = global_params["activation"]
        self._has_se = (global_params["use_se"]
                        and self._block_args["se_ratio"] is not None
                        and 0 < self._block_args["se_ratio"] <= 1)
        self._clip_projection_output = global_params["clip_projection_output"]
        self.endpoints = None
        if self._block_args["condconv"]:
            raise ValueError('Condconv is not supported')
        # Builds the block according to arguments.
        self._build()

    @property
    def block_args(self):
        return self._block_args

    def get_conv_name(self):
        name = 'conv2d' + ("" if not next(self.conv_id)
                           else '_' + str(next(self.conv_id) // 2))
        return name

    def get_batch_norm_name(self):
        name = 'batch_normalization' + (
            "" if not next(self.batch_norm_id)
            else '_' + str(next(self.batch_norm_id) // 2))
        return name

    def build_se(self, filters):
        """
        Builds Squeeze-and-excitation block.

        # Arguments
            filters: Int, output filters of squeeze and excite block.
        """
        if self._has_se:
            input_filters = self._block_args["input_filters"]
            se_ratio = self._block_args["se_ratio"]
            num_reduced_filters = max(1, int(input_filters * se_ratio))
            self._se = SE(self._global_params, num_reduced_filters, filters,
                          name='se')
        else:
            self._se = None

    def build_super_pixel(self):
        if self._block_args["super_pixel"] == 1:
            self.super_pixel = SuperPixel(
                self._block_args, self._global_params, name='super_pixel')
        else:
            self.super_pixel = None

    def build_fused_conv(self, filters):
        """
        # Arguments
            filters: Int, output filters of fused conv block.
        """
        kernel_size = self._block_args["kernel_size"]
        if self._block_args["fused_conv"]:
            # Fused expansion phase. Called if using fused convolutions.
            self._fused_conv = tf.keras.layers.Conv2D(
                filters, [kernel_size, kernel_size],
                self._block_args["strides"], 'same', 'channels_last',
                (1, 1), 1, None, False, conv_kernel_initializer,
                name=self.get_conv_name())
        else:
            # Expansion phase.
            # Called if not using fused convolution and expansion
            # phase is necessary.
            if self._block_args["expand_ratio"] != 1:
                self._expand_conv = Conv2D(
                    filters, [1, 1], [1, 1], 'same', 'channels_last',
                    (1, 1), 1, None, False, conv_kernel_initializer,
                    name=self.get_conv_name())
                self._batch_norm0 = self._batch_norm(
                    -1, 0.99, 1e-3, name=self.get_batch_norm_name())
            # Depth-wise convolution phase.
            # Called if not using fused convolutions.
            self._depthwise_conv = DepthwiseConv2D(
                [kernel_size, kernel_size], self._block_args["strides"],
                'same', 1, 'channels_last', (1, 1), None, False,
                conv_kernel_initializer, name='depthwise_conv2d')

    def _build(self):
        """Builds block according to the arguments."""
        self.conv_id = itertools.count(0)
        self.batch_norm_id = itertools.count(0)
        self.build_super_pixel()
        block_input_filters = self._block_args["input_filters"]
        block_expand_ratio = self._block_args["expand_ratio"]
        filters = block_input_filters * block_expand_ratio
        self.build_fused_conv(filters)
        self._batch_norm1 = self._batch_norm(
            -1, 0.99, 1e-3, name=self.get_batch_norm_name())
        self.build_se(filters)
        # Output phase.
        filters = self._block_args["output_filters"]
        self._project_conv = Conv2D(
            filters, [1, 1], [1, 1], 'same', 'channels_last', (1, 1), 1, None,
            False, conv_kernel_initializer, name=self.get_conv_name())
        self._batch_norm2 = self._batch_norm(
            -1, 0.99, 1e-3, name=self.get_batch_norm_name())

    def call(self, tensor, training, survival_rate):
        """Implementation of call().

        # Arguments
            tensor: Tensor, the inputs tensor.
            training: boolean, whether the model is constructed for training.
            survival_rate: float, between 0 to 1, drop connect rate.

        # Returns
            A output tensor.
        """
        def _call(tensor):
            """
            # Arguments
                tensor: Tensor, the inputs tensor.

            # Returns
                tensor: Tensor, output Mobile Bottleneck block processed.
            """
            print('Block %s input shape: %s', self.name, tensor.shape)
            x = tensor

            # creates conv 2x2 kernel
            if self.super_pixel:
                x = self.super_pixel(x, training)
                print('SuperPixel %s: %s', self.name, x.shape)

            if self._block_args["fused_conv"]:
                # If use fused mbconv, skip expansion and use regular conv.
                x = self._batch_norm1(self._fused_conv(x), training=training)
                x = get_activation(x, self._activation)
                print('Conv2D shape: %s', x.shape)
            else:
                # Otherwise, first apply expansion
                # and then apply depthwise conv.
                if self._block_args["expand_ratio"] != 1:
                    x = self._batch_norm0(
                        self._expand_conv(x), training=training)
                    x = get_activation(x, self._activation)
                    print('Expand shape: %s', x.shape)

                x = self._batch_norm1(
                    self._depthwise_conv(x), training=training)
                x = get_activation(x, self._activation)
                print('DWConv shape: %s', x.shape)

            if self._se:
                x = self._se(x)

            self.endpoints = {'expansion_output': x}

            x = self._batch_norm2(self._project_conv(x), training=training)
            # Add identity so that quantization-aware
            # training can insert quantization ops correctly.
            x = tf.identity(x)
            if self._clip_projection_output:
                x = tf.clip_by_value(x, -6, 6)
            if self._block_args["id_skip"]:
                if (all(s == 1 for s in self._block_args["strides"])
                        and self._block_args["input_filters"]
                        == self._block_args["output_filters"]):
                    # Apply only if skip connection presents.
                    if survival_rate:
                        x = get_drop_connect(x, training, survival_rate)
                    x = tf.add(x, tensor)
            print('Project shape: %s', x.shape)
            return x

        return _call(tensor)


class MBConvBlockWithoutDepthwise(MBConvBlock):
    pass


def round_filters(filters, global_params, skip=False):
    """Round number of filters based on depth multiplier.
    # Arguments
        filters: Int, filters to be rounded based on depth multiplier.
        global_params: GlobalParams, a set of global parameters.
        skip: Bool, skip rounding filters based on multiplier.

    # Returns
        new_filters: Int, rounded filters based on depth multiplier.
    """
    multiplier = global_params["width_coefficient"]
    divisor = global_params["depth_divisor"]
    min_depth = global_params["min_depth"]
    if skip or not multiplier:
        return filters
    filters = filters * multiplier
    min_depth = min_depth or divisor
    new_filters = max(
        min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters = new_filters + divisor
    new_filters = int(new_filters)
    return new_filters


def round_repeats(repeats, global_params, skip=False):
    """Round number of filters based on depth multiplier.

    # Arguments
        repeats: Int, number of repeats of multiplier blocks.
        global_params: GlobalParams, a set of global parameters.
        skip: Bool, skip rounding filters based on multiplier.

    # Returns
        new_repeats: Int, repeats of blocks based on multiplier.
    """
    multiplier = global_params["depth_coefficient"]
    if skip or not multiplier:
        return repeats
    new_repeats = int(math.ceil(multiplier * repeats))
    return new_repeats


class Stem(Layer):
    """Stem layer at the beginning of the network."""

    def __init__(self, global_params, stem_filters, name=None):
        """
        # Arguments
            global_params: GlobalParams, a set of global parameters.
            stem_filters: Int, filter count for the stem block.
            name: String, layer name.
        """
        super().__init__(name=name)
        filters = round_filters(
            stem_filters, global_params, global_params["fix_head_stem"])
        self._conv_stem = Conv2D(filters, [3, 3], [2, 2], 'same',
                                 'channels_last', (1, 1), 1, None, False,
                                 conv_kernel_initializer)
        self._batch_norm = global_params["batch_norm"](-1, 0.99, 1e-3)
        self._activation = global_params["activation"]

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

    def __init__(self, global_params, name=None):
        """
        # Arguments
            global_params: GlobalParams, a set of global parameters.
        """
        super().__init__(name=name)

        self.endpoints = {}
        self._global_params = global_params
        conv_filters = round_filters(
            1280, global_params, global_params["fix_head_stem"])
        self._conv_head = Conv2D(
            conv_filters, [1, 1], [1, 1], 'same', 'channels_last', (1, 1), 1,
            None, False, conv_kernel_initializer, name='conv2d')
        self._batch_norm = global_params["batch_norm"](-1, 0.99, 1e-3)
        self._activation = global_params["activation"]
        self._avg_pooling = GlobalAveragePooling2D(data_format='channels_last')
        if global_params["num_classes"]:
            self._fully_connected = Dense(global_params["num_classes"],
                             kernel_initializer=dense_kernel_initializer)
        else:
            self._fully_connected = None
        if global_params["dropout_rate"] > 0:
            self._dropout = tf.keras.layers.Dropout(
                global_params["dropout_rate"])
        else:
            self._dropout = None
        self.h_axis, self.w_axis = ([1, 2])

    def call(self, tensor, training, pool_features):
        """Call the head layer.
        # Arguments
            tensor: Tensor, the inputs tensor.
            training: boolean, whether the model is constructed for training.
            pool_features: Bool, flag to decide feature pooling from tensor.
        """
        outputs = self._batch_norm(self._conv_head(tensor), training=training)
        outputs = get_activation(outputs, self._activation)
        self.endpoints['head_1x1'] = outputs

        if self._global_params["local_pooling"]:
            shape = outputs.get_shape().as_list()
            kernel_size = [1, shape[self.h_axis], shape[self.w_axis], 1]
            outputs = tf.nn.avg_pool(
                outputs, kernel_size, [1, 1, 1, 1], 'VALID')
            self.endpoints['pooled_features'] = outputs
            if not pool_features:
                if self._dropout:
                    outputs = self._dropout(outputs, training=training)
                self.endpoints['global_pool'] = outputs
                if self._fully_connected:
                    outputs = tf.squeeze(outputs, [self.h_axis, self.w_axis])
                    outputs = self._fully_connected(outputs)
                self.endpoints['head'] = outputs
        else:
            outputs = self._avg_pooling(outputs)
            self.endpoints['pooled_features'] = outputs
            if not pool_features:
                if self._dropout:
                    outputs = self._dropout(outputs, training=training)
                self.endpoints['global_pool'] = outputs
                if self._fully_connected:
                    outputs = self._fully_connected(outputs)
                self.endpoints['head'] = outputs
        return outputs


class Model(tf.keras.Model):
    """A class implementing tf.keras.Model for EfficientNet."""

    def __init__(self, blocks_args=None, global_params=None, name=None):
        """Initializes an 'Model' instance.

        # Arguments
            blocks_args: Dictionary of BlockArgs to construct block modules.
            global_params: GlobalParams, a set of global parameters.
            name: A string of layer name.

        # Raises
            ValueError: when blocks_args is not specified as list.
        """
        super().__init__(name=name)

        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._activation = global_params["activation"]
        self._batch_norm = global_params["batch_norm"]
        self._fix_head_stem = global_params["fix_head_stem"]
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

    def update_block_repeats(self, block_args, block_num):
        """Update block repeats based on depth multiplier.
        # Arguments
            block_args: Dictionary, A list of BlockArgs to construct
            block modules.
            block_num: Int, Block index.

        # Returns
            block_args: Dictionary, A list of BlockArgs to construct
            block modules with updated repeats of block.
        """
        blocks_repeat_limit = len(self._blocks_args) - 1
        block_flag = (block_num == 0 or block_num == blocks_repeat_limit)
        if self._fix_head_stem and block_flag:
            repeats = block_args["num_repeat"]
        else:
            repeats = round_repeats(
                block_args["num_repeat"], self._global_params)
        params = {"num_repeat": repeats}
        block_args.update(params)
        return block_args

    def add_block_repeats(self, block_args):
        """
        # Arguments
            block_args: Dictionary, A list of BlockArgs to construct
            block modules.
        """
        conv_block = self._get_conv_block(block_args["conv_type"])
        if block_args["num_repeat"] > 1:
            # rest of blocks with the same block_arg
            block_args.update({
                "input_filters": block_args["output_filters"],
                "strides": [1, 1]})
        for _ in range(block_args["num_repeat"] - 1):
            self._blocks.append(conv_block(
                block_args.copy(), self._global_params,
                name=self.get_block_name()))

    def update_block_depth(self, block_args,
                           block_strides_1, block_strides_2):
        """
        # Arguments
            block_args: Dictionary, A list of BlockArgs to construct
            block modules.
            block_strides_1: Int, x stride of convolution layers.
            block_strides_2: Int, y stride of convolution layers.

        # Returns
            block_args: Dictionary, A list of BlockArgs to construct
            block modules.
        """
        depth_factor = int(4 / block_strides_1 / block_strides_2)
        if depth_factor > 1:
            kernel_size = (block_args["kernel_size"] + 1) // 2
        else:
            kernel_size = block_args["kernel_size"]
        filter_depth_input = block_args["input_filters"] * depth_factor
        filter_depth_output = block_args["output_filters"] * depth_factor
        params = {"input_filters": filter_depth_input,
                  "output_filters": filter_depth_output,
                  "kernel_size": kernel_size}
        block_args = block_args.update(params)
        return block_args

    def add_stride2_block(self, block_args, input_filters, output_filters):
        """
        # Arguments
            block_args: Dictionary, A list of BlockArgs to construct
            block modules.
            input_filters: Int, Input filters for the blocks to construct.
            output_filters: Int, Output filters for the blocks to construct.

        # Returns
            block_args: Dictionary, A list of BlockArgs to construct
            block modules.
        """
        conv_block = self._get_conv_block(block_args["conv_type"])
        kernel_size = block_args["kernel_size"]
        block_args.update({"strides": [1, 1]})
        self._blocks.append(conv_block(
            block_args.copy(), self._global_params,
            name=self.get_block_name()))
        block_args.update({
            "super_pixel": 0, "input_filters": input_filters,
            "output_filters": output_filters,
            "kernel_size": kernel_size})
        return block_args

    def build_super_pixel_blocks(self, block_args,
                                 input_filters, output_filters):
        """
        # Arguments
            block_args: Dictionary, A list of BlockArgs to construct
            block modules.
            input_filters: Int, Input filters for the blocks to construct.
            output_filters: Int, Output filters for the blocks to construct.

        # Returns
            block_args: Dictionary, A list of BlockArgs to construct
            block modules with updated parameters.
        """
        conv_block = self._get_conv_block(block_args["conv_type"])
        # if superpixel, adjust filters, kernels, and strides.
        block_strides_1 = block_args["strides"][0]
        block_strides_2 = block_args["strides"][1]
        block_args = self.update_block_depth(
            block_args, block_strides_1, block_strides_2)
        # if the first block has stride-2 and superpixel transformation
        if block_strides_1 == 2 and block_strides_2 == 2:
            block_args = self.add_stride2_block(
                block_args, input_filters, output_filters)
        elif block_args["super_pixel"] == 1:
            self._blocks.append(conv_block(
                block_args.copy(), self._global_params,
                name=self.get_block_name())
            )
            block_args.update({"super_pixel": 2})
        else:
            self._blocks.append(conv_block(
                block_args.copy(), self._global_params,
                name=self.get_block_name())
            )
        return block_args

    def build_blocks(self, block_args, input_filters, output_filters):
        """
        # Arguments
            block_args: Dictionary, A list of BlockArgs to construct
            block modules.
            input_filters: Int, Input filters for the blocks to construct.
            output_filters: Int, Output filters for the blocks to construct.
        """

        # The first block needs to take care of stride
        # and filter size increase.
        conv_block = self._get_conv_block(block_args["conv_type"])
        if not block_args["super_pixel"]:  # no super_pixel at all
            self._blocks.append(conv_block(
                block_args.copy(), self._global_params,
                name=self.get_block_name()))
        else:
            block_args = self.build_super_pixel_blocks(
                block_args, input_filters, output_filters)
        self.add_block_repeats(block_args)

    def update_filters(self, block_args):
        """Update block input and output filters based on depth multiplier.
        # Arguments
            block_args: Dictionary, A list of BlockArgs to construct
            block modules.

        # Returns
            block_args: Dictionary, A list of BlockArgs to construct
            block modules with updated filter counts.
        """
        input_filters = round_filters(
            block_args["input_filters"], self._global_params)
        output_filters = round_filters(
            block_args["output_filters"], self._global_params)
        block_args.update({"input_filters": input_filters,
                           "output_filters": output_filters})
        return block_args

    def _build(self):
        """Builds a model."""
        self._blocks = []

        # Stem part.
        self._stem = Stem(self._global_params,
                          self._blocks_args[0]["input_filters"])
        self.block_id = itertools.count(0)

        # Builds blocks.
        for block_num, block_args in enumerate(self._blocks_args):
            assert block_args["num_repeat"] > 0
            assert block_args["super_pixel"] in [0, 1, 2]
            block_args = self.update_filters(block_args)
            block_args = self.update_block_repeats(block_args, block_num)
            self.build_blocks(block_args, block_args["input_filters"],
                              block_args["output_filters"])
        # Head part.
        self._head = Head(self._global_params)

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
        reduction_idx = 0

        # Calls Stem layers
        outputs = self._stem(tensor, training)
        print('Built stem %s : %s', self._stem.name, outputs.shape)
        self.endpoints['stem'] = outputs

        # Call blocks
        for idx, block in enumerate(self._blocks):
            is_reduction = False
            # reduction flag for blocks after the stem layer
            # If the first block has super-pixel (space-to-depth)
            # layer, then stem is the first reduction point
            if (block.block_args["super_pixel"] == 1 and idx == 0):
                reduction_idx = reduction_idx + 1
                self.endpoints['reduction_%s' % reduction_idx] = outputs

            elif ((idx == len(self._blocks) - 1) or
                    self._blocks[idx + 1].block_args["strides"][0] > 1):
                is_reduction = True
                reduction_idx = reduction_idx + 1

            survival_rate = self._global_params["survival_rate"]
            if survival_rate:
                drop_rate = 1 - survival_rate
                survival_rate = 1 - drop_rate * float(idx) / len(self._blocks)
            outputs = block(
                outputs, training=training, survival_rate=survival_rate)
            self.endpoints['block_%s' % idx] = outputs
            if is_reduction:
                self.endpoints['reduction_%s' % reduction_idx] = outputs
            if block.endpoints:
                for k, v in six.iteritems(block.endpoints):
                    self.endpoints['block_%s/%s' % (idx, k)] = v
                    if is_reduction:
                        self.endpoints[
                            'reduction_%s/%s' % (reduction_idx, k)
                        ] = v
        self.endpoints['features'] = outputs

        if not return_base:
            # Calls final layers and returns logits.
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
