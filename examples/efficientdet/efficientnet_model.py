"""Instantiates the EfficientNet architecture using given scaling coefficients.
Reference:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
 https://arxiv.org/abs/1905.11946) (ICML 2019)
"""
import collections
import itertools
import math
import numpy as np
import six
import tensorflow as tf
from tensorflow.keras.layers import Layer, \
    Conv2D, \
    GlobalAveragePooling2D, \
    Dense, \
    DepthwiseConv2D
from utils import get_activation_fn, \
    get_drop_connect


GlobalParams = collections.namedtuple(
    'GlobalParams',
    [
        'dropout_rate',
        'data_format',
        'num_classes',
        'width_coefficient',
        'depth_coefficient',
        'depth_divisor',
        'min_depth',
        'survival_prob',
        'act_fn',
        'batch_norm',
        'use_se',
        'local_pooling',
        'condconv_num_experts',
        'clip_projection_output',
        'blocks_args',
        'fix_head_stem'
    ]
)
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple(
    'BlockArgs',
    [
        'kernel_size',
        'num_repeat',
        'input_filters',
        'output_filters',
        'expand_ratio',
        'id_skip',
        'strides',
        'se_ratio',
        'conv_type',
        'fused_conv',
        'super_pixel',
        'condconv'
    ]
)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


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
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(shape,
                            mean=0.0,
                            stddev=np.sqrt(2.0 / fan_out),
                            dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.
    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
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
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


def superpixel_kernel_initializer(shape,
                                  dtype='float32',
                                  partition_info=None):
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
    i = np.arange(2)
    j = np.arange(2)
    k = np.arange(depth)
    mesh = np.array(np.meshgrid(i, j, k)).T.reshape(-1, 3).T
    filters[mesh[0], mesh[1], mesh[2], 4 * mesh[2] + 2 * mesh[0] + mesh[1]] = 1
    return filters


class SE(Layer):
    """Squeeze-and-excitation layer."""
    def __init__(self, global_params, se_filters, output_filters, name=None):
        super().__init__(name=name)

        self._local_pooling = global_params.local_pooling
        self._act_fn = global_params.act_fn

        # Squeeze and Excitation layer.
        self._se_reduce = Conv2D(
            se_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format='channels_last',
            use_bias=True,
            name='conv2d'
        )
        self._se_expand = Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format='channels_last',
            use_bias=True,
            name='conv2d_1',
        )

    def call(self, tensor):
        h_axis, w_axis = [1, 2]
        if self._local_pooling:
            se_tensor = tf.nn.avg_pool(
                tensor,
                ksize=[1, tensor.shape[h_axis], tensor.shape[w_axis], 1],
                strides=[1, 1, 1, 1],
                padding='VALID'
            )
        else:
            se_tensor = tf.reduce_mean(tensor, [h_axis, w_axis], keepdims=True)
        se_tensor = self._se_expand(get_activation_fn(
            self._se_reduce(se_tensor),
            self._act_fn)
        )
        print('Built SE %s : %s', self.name, se_tensor.shape)
        return tf.sigmoid(se_tensor) * tensor


class SuperPixel(Layer):
    """Super pixel layer."""

    def __init__(self, block_args, global_params, name=None):
        super().__init__(name=name)
        self._superpixel = Conv2D(
            block_args.input_filters,
            kernel_size=[2, 2],
            strides=[2, 2],
            kernel_initializer=superpixel_kernel_initializer,
            padding='same',
            data_format='channels_last',
            use_bias=False,
            name='conv2d'
        )
        self._batch_norm_superpixel = global_params.batch_norm(axis=-1,
                                                               momentum=0.99,
                                                               epsilon=1e-3)
        self._act_fn = global_params.act_fn

    def call(self, tensor, training):
        out = self._superpixel(tensor)
        out = self._batch_norm_superpixel(out, training)
        out = get_activation_fn(out, self._act_fn)
        return out


class MBConvBlock(Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.
    # Attributes
        endpoints: dict. A list of internal tensors.
    """

    def __init__(self, block_args, global_params, name=None):
        """Initializes a MBConv block.

        # Arguments
            block_args: BlockArgs, arguments to create a Block.
            global_params: GlobalParams, a set of global parameters.
            name: layer name.
        """
        super().__init__(name=name)
        self._block_args = block_args
        self._global_params = global_params
        self._local_pooling = global_params.local_pooling
        self._batch_norm = global_params.batch_norm
        self._condconv_num_experts = global_params.condconv_num_experts
        self._act_fn = global_params.act_fn
        self._has_se = (global_params.use_se
                        and self._block_args.se_ratio is not None
                        and 0 < self._block_args.se_ratio <= 1
                        )
        self._clip_projection_output = global_params.clip_projection_output
        self.endpoints = None
        if self._block_args.condconv:
            raise ValueError('Condconv is not supported')

        # Builds the block according to arguments.
        self._build()

    @property
    def block_args(self):
        return self._block_args

    @staticmethod
    def get_conv_name():
        cid = itertools.count(0)
        name = 'conv2d' + (
            "" if not next(cid) else '_' + str(next(cid) // 2)
        )
        return name

    @staticmethod
    def get_bn_name():
        bid = itertools.count(0)
        name = 'batch_normalization' + (
            "" if not next(bid) else '_' + str(next(bid) // 2)
        )
        return name

    def _build(self):
        """Builds block according to the arguments."""
        if self._block_args.super_pixel == 1:
            self.super_pixel = SuperPixel(
                self._block_args, self._global_params, name='super_pixel'
            )
        else:
            self.super_pixel = None

        block_input_filters = self._block_args.input_filters
        block_expand_ratio = self._block_args.expand_ratio
        filters = block_input_filters * block_expand_ratio
        kernel_size = self._block_args.kernel_size

        if self._block_args.fused_conv:
            # Fused expansion phase. Called if using fused convolutions.
            self._fused_conv = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=[kernel_size, kernel_size],
                strides=self._block_args.strides,
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=False,
                name=self.get_conv_name(),
            )
        else:
            # Expansion phase.
            # Called if not using fused convolution and expansion
            # phase is necessary.
            if self._block_args.expand_ratio != 1:
                self._expand_conv = Conv2D(
                    filters=filters,
                    kernel_size=[1, 1],
                    strides=[1, 1],
                    kernel_initializer=conv_kernel_initializer,
                    padding='same',
                    data_format='channels_last',
                    use_bias=False,
                    name=self.get_conv_name()
                )
                self._batch_norm0 = self._batch_norm(axis=-1,
                                                     momentum=0.99,
                                                     epsilon=1e-3,
                                                     name=self.get_bn_name()
                                                     )
            # Depth-wise convolution phase.
            # Called if not using fused convolutions.
            self._depthwise_conv = DepthwiseConv2D(
                kernel_size=[kernel_size, kernel_size],
                strides=self._block_args.strides,
                depthwise_initializer=conv_kernel_initializer,
                padding='same',
                data_format='channels_last',
                use_bias=False,
                name='depthwise_conv2d',
            )
        self._batch_norm1 = self._batch_norm(axis=-1,
                                             momentum=0.99,
                                             epsilon=1e-3,
                                             name=self.get_bn_name()
                                             )

        if self._has_se:
            num_reduced_filters = max(
                1,
                int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se = SE(self._global_params,
                          num_reduced_filters,
                          filters,
                          name='se')
        else:
            self._se = None

        # Output phase.
        filters = self._block_args.output_filters
        self._project_conv = Conv2D(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format='channels_last',
            use_bias=False,
            name=self.get_conv_name()
        )
        self._batch_norm2 = self._batch_norm(axis=-1,
                                             momentum=0.99,
                                             epsilon=1e-3,
                                             name=self.get_bn_name())

    def call(self, tensor, training, survival_prob):
        """Implementation of call().
        # Arguments
            inputs: the inputs tensor.
            training: boolean, whether the model is constructed for training.
            survival_prob: float, between 0 to 1, drop connect rate.
        # Returns
            A output tensor.
        """
        def _call(tensor):
            print('Block %s input shape: %s', self.name, tensor.shape)
            x = tensor

            # creates conv 2x2 kernel
            if self.super_pixel:
                x = self.super_pixel(x, training)
                print('SuperPixel %s: %s', self.name, x.shape)

            if self._block_args.fused_conv:
                # If use fused mbconv, skip expansion and use regular conv.
                x = self._batch_norm1(self._fused_conv(x), training=training)
                x = get_activation_fn(x, self._act_fn)
                print('Conv2D shape: %s', x.shape)
            else:
                # Otherwise, first apply expansion
                # and then apply depthwise conv.
                if self._block_args.expand_ratio != 1:
                    x = self._batch_norm0(self._expand_conv(x),
                                          training=training)
                    x = get_activation_fn(x, self._act_fn)
                    print('Expand shape: %s', x.shape)

                x = self._batch_norm1(self._depthwise_conv(x),
                                      training=training)
                x = get_activation_fn(x, self._act_fn)
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

            if self._block_args.id_skip:
                if (
                    all(s == 1 for s in self._block_args.strides)
                    and self._block_args.input_filters
                    == self._block_args.output_filters
                ):
                    # Apply only if skip connection presents.
                    if survival_prob:
                        x = get_drop_connect(x, training, survival_prob)
                    x = tf.add(x, tensor)
            print('Project shape: %s', x.shape)
            return x

        return _call(tensor)


class MBConvBlockWithoutDepthwise(MBConvBlock):
    pass


def round_filters(filters, global_params, skip=False):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if skip or not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth,
                      int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params, skip=False):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if skip or not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class Stem(Layer):
    """Stem layer at the beginning of the network."""

    def __init__(self, global_params, stem_filters, name=None):
        super().__init__(name=name)
        self._conv_stem = Conv2D(
            filters=round_filters(
                stem_filters, global_params, global_params.fix_head_stem
            ),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format='channels_last',
            use_bias=False,
        )
        self._batch_norm = global_params.batch_norm(axis=-1,
                                                    momentum=0.99,
                                                    epsilon=1e-3)
        self._act_fn = global_params.act_fn

    def call(self, tensor, training):
        out = self._batch_norm(self._conv_stem(tensor,
                                               training=training),
                               training=training)
        out = get_activation_fn(out, self._act_fn)
        return out


class Head(Layer):
    """Head layer for network outputs."""

    def __init__(self, global_params, name=None):
        super().__init__(name=name)

        self.endpoints = {}
        self._global_params = global_params

        self._conv_head = Conv2D(
            filters=round_filters(1280,
                                  global_params,
                                  global_params.fix_head_stem),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format='channels_last',
            use_bias=False,
            name='conv2d'
        )

        self._batch_norm = global_params.batch_norm(axis=-1,
                                                    momentum=0.99,
                                                    epsilon=1e-3)
        self._act_fn = global_params.act_fn
        self._avg_pooling = GlobalAveragePooling2D(data_format='channels_last')
        if global_params.num_classes:
            self._fc = Dense(global_params.num_classes,
                             kernel_initializer=dense_kernel_initializer)
        else:
            self._fc = None

        if global_params.dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(global_params.dropout_rate)
        else:
            self._dropout = None

        self.h_axis, self.w_axis = ([1, 2])

    def call(self, tensor, training, pooled_features_only):
        """Call the head layer."""
        outputs = self._batch_norm(self._conv_head(tensor), training=training)
        outputs = get_activation_fn(outputs, self._act_fn)
        self.endpoints['head_1x1'] = outputs

        if self._global_params.local_pooling:
            shape = outputs.get_shape().as_list()
            kernel_size = [1, shape[self.h_axis], shape[self.w_axis], 1]
            outputs = tf.nn.avg_pool(
                outputs,
                ksize=kernel_size,
                strides=[1, 1, 1, 1],
                padding='VALID'
            )
            self.endpoints['pooled_features'] = outputs
            if not pooled_features_only:
                if self._dropout:
                    outputs = self._dropout(outputs, training=training)
                self.endpoints['global_pool'] = outputs
                if self._fc:
                    outputs = tf.squeeze(outputs, [self.h_axis, self.w_axis])
                    outputs = self._fc(outputs)
                self.endpoints['head'] = outputs
            else:
                outputs = self._avg_pooling(outputs)
                self.endpoints['pooled_features'] = outputs
                if not pooled_features_only:
                    if self._dropout:
                        outputs = self._dropout(outputs, training=training)
                    self.endpoints['global_pool'] = outputs
                    if self._fc:
                        outputs = self._fc(outputs)
                    self.endpoints['head'] = outputs
            return outputs


class Model(tf.keras.Model):
    """A class implementing tf.keras.Model for EfficientNet."""

    def __init__(self, blocks_args=None, global_params=None, name=None):
        """Initializes an 'Model' instance.
        # Arguments
            blocks_args: A list of BlockArgs to construct block modules.
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
        self._act_fn = global_params.act_fn
        self._batch_norm = global_params.batch_norm
        self._fix_head_stem = global_params.fix_head_stem
        self.endpoints = None

        self._build()

    def _get_conv_block(self, conv_type):
        conv_block_map = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}
        return conv_block_map[conv_type]

    @staticmethod
    def get_block_name():
        block_id = itertools.count(0)
        name = 'blocks_%d' % next(block_id)
        return name

    def _build(self):
        """Builds a model."""
        self._blocks = []

        # Stem part.
        self._stem = Stem(self._global_params,
                          self._blocks_args[0].input_filters)

        # Builds blocks.
        for i, block_args in enumerate(self._blocks_args):
            assert block_args.num_repeat > 0
            assert block_args.super_pixel in [0, 1, 2]
            # Update block input and output filters based on depth multiplier.
            input_filters = round_filters(block_args.input_filters,
                                          self._global_params)

            output_filters = round_filters(
                block_args.output_filters, self._global_params
            )
            kernel_size = block_args.kernel_size
            if self._fix_head_stem \
                    and (i == 0 or i == len(self._blocks_args) - 1):
                repeats = block_args.num_repeat
            else:
                repeats = round_repeats(block_args.num_repeat,
                                        self._global_params)
            block_args = block_args._replace(
                input_filters=input_filters,
                output_filters=output_filters,
                num_repeat=repeats,
            )

            # The first block needs to take care of stride
            # and filter size increase.
            conv_block = self._get_conv_block(block_args.conv_type)
            if not block_args.super_pixel:  # no super_pixel at all
                self._blocks.append(
                    conv_block(block_args,
                               self._global_params,
                               name=self.get_block_name())
                )
            else:
                # if superpixel, adjust filters, kernels, and strides.
                block_strides_1 = block_args.strides[0]
                block_strides_2 = block_args.strides[1]
                depth_factor = int(4 / block_strides_1 / block_strides_2)
                block_args = block_args._replace(
                    input_filters=block_args.input_filters * depth_factor,
                    output_filters=block_args.output_filters * depth_factor,
                    kernel_size=(
                        (block_args.kernel_size + 1) // 2
                        if depth_factor > 1
                        else block_args.kernel_size
                    ),
                )
                # if the first block has stride-2 and superpixel transformation
                if block_args.strides[0] == 2 and block_args.strides[1] == 2:
                    block_args = block_args._replace(strides=[1, 1])
                    self._blocks.append(
                        conv_block(block_args,
                                   self._global_params,
                                   name=self.get_block_name())
                    )
                    block_args = block_args._replace(  # sp stops at stride-2
                        super_pixel=0,
                        input_filters=input_filters,
                        output_filters=output_filters,
                        kernel_size=kernel_size,
                    )
                elif block_args.super_pixel == 1:
                    self._blocks.append(
                        conv_block(block_args,
                                   self._global_params,
                                   name=self.get_block_name())
                    )
                    block_args = block_args._replace(super_pixel=2)
                else:
                    self._blocks.append(
                        conv_block(block_args,
                                   self._global_params,
                                   name=self.get_block_name())
                    )
            if block_args.num_repeat > 1:
                # rest of blocks with the same block_arg
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1]
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    conv_block(block_args,
                               self._global_params,
                               name=self.get_block_name())
                )

        # Head part.
        self._head = Head(self._global_params)

    def call(self,
             tensor,
             training,
             features_only=None,
             pooled_features_only=False):
        """Implementation of call().
        # Arguments
            tensor: input tensors.
            training: boolean, whether the model is constructed for training.
            features_only: build the base feature network only.
            pooled_features_only: build the base network for
            features extraction (after 1x1 conv layer and global
             pooling, but before dropout and fc head).
        # Returns
          output tensors.
        """
        outputs = None
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
            if (block.block_args.super_pixel == 1 and idx == 0):
                reduction_idx += 1
                self.endpoints['reduction_%s' % reduction_idx] = outputs

            elif ((idx == len(self._blocks) - 1) or
                    self._blocks[idx + 1].block_args.strides[0] > 1):
                is_reduction = True
                reduction_idx += 1

            survival_prob = self._global_params.survival_prob
            if survival_prob:
                drop_rate = 1 - survival_prob
                survival_prob = 1 - drop_rate * float(idx) / len(self._blocks)
            outputs = block(outputs,
                            training=training,
                            survival_prob=survival_prob)
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

        if not features_only:
            # Calls final layers and returns logits.
            outputs = self._head(outputs, training, pooled_features_only)
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
