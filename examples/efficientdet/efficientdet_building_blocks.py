import functools

import tensorflow as tf
from config import get_fpn_configuration
from tensorflow.keras.layers import Layer


class ResampleFeatureMap(Layer):
    """Resample feature maps for downsampling or upsampling
     to create coarser or finer additional feature maps."""
    def __init__(self,
                 feature_level,
                 target_num_channels,
                 apply_bn,
                 conv_after_downsample,
                 data_format='channels_last',
                 pooling_type=None,
                 upsampling_type=None,
                 name='resample_p0'):

        super().__init__(name=name)
        self.apply_bn = apply_bn
        self.data_format = data_format
        self.target_num_channels = target_num_channels
        self.feature_level = feature_level
        self.conv_after_downsample = conv_after_downsample
        self.pooling_type = pooling_type or 'max'
        self.upsampling_type = upsampling_type or 'nearest'
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.99,
                                                     epsilon=1e-3)
        self.conv2d = tf.keras.layers.Conv2D(self.target_num_channels,
                                             (1, 1),
                                             padding='same',
                                             data_format=self.data_format,
                                             name='conv2d')

    def _pool2d(self, inputs, height, width, target_height, target_width):
        """Pool the inputs to target height and width."""
        height_stride_size = int((height - 1) // target_height + 1)
        width_stride_size = int((width - 1) // target_width + 1)
        if self.pooling_type == 'max':
            return tf.keras.layers.MaxPooling2D(
                pool_size=[height_stride_size + 1, width_stride_size + 1],
                strides=[height_stride_size, width_stride_size],
                padding='SAME',
                data_format=self.data_format)(inputs)
        if self.pooling_type == 'avg':
            return tf.keras.layers.AveragePooling2D(
                pool_size=[height_stride_size + 1, width_stride_size + 1],
                strides=[height_stride_size, width_stride_size],
                padding='SAME',
                data_format=self.data_format)(inputs)
        raise ValueError('Unsupported pooling type {}.'.format(
            self.pooling_type))

    def _upsample2d(self, inputs, target_height, target_width):
        return tf.cast(
            tf.compat.v1.image.resize_nearest_neighbor(
                tf.cast(inputs, tf.float32), [target_height, target_width]),
            inputs.dtype)

    def _maybe_apply_1x1(self, feature, training, num_channels):
        """Apply 1x1 conv to change layer width if necessary."""
        if num_channels != self.target_num_channels:
            feature = self.conv2d(feature)
            if self.apply_bn:
                feature = self.bn(feature, training=training)
        return feature

    def call(self, feature, training, all_features):
        hwc_idx = (2, 3, 1) if self.data_format == 'channels_first' \
            else (1, 2, 3)
        height, width, num_channels = \
            [feature.shape.as_list()[i] for i in hwc_idx]
        if all_features:
            target_feature_shape = \
                all_features[self.feature_level].shape.as_list()
            target_height, target_width, _ = \
                [target_feature_shape[i] for i in hwc_idx]
        else:
            # Default to downsampling if all_features is empty.
            target_height, target_width = (height + 1) // 2, (width + 1) // 2

        # If conv_after_downsample is True, when downsampling, apply 1x1 after
        # downsampling for efficiency.
        if height > target_height and width > target_width:
            if not self.conv_after_downsample:
                feature = self._maybe_apply_1x1(feature,
                                                training,
                                                num_channels)
            feature = self._pool2d(feature,
                                   height,
                                   width,
                                   target_height,
                                   target_width)
            if self.conv_after_downsample:
                feature = self._maybe_apply_1x1(feature,
                                                training,
                                                num_channels)
        elif height <= target_height and width <= target_width:
            feature = self._maybe_apply_1x1(feature, training, num_channels)
            if height < target_height or width < target_width:
                feature = self._upsample2d(feature,
                                           target_height,
                                           target_width)
        else:
            raise ValueError('Incompatible Resampling : feature shape {}x{} \
            target_shape: {}x{}'.format(height,
                                        width,
                                        target_height,
                                        target_width))

        return feature


class FPNCells(Layer):
    """Set of FPN Cells."""
    def __init__(self, config, name='fpn_cells'):
        super().__init__(name=name)
        self.config = config
        self.fpn_config = get_fpn_configuration(config['fpn_name'],
                                                config['min_level'],
                                                config['max_level'],
                                                config['fpn_weight_method']
                                                )
        self.cells = [FPNCell(self.config, name='cell_%d' % repeats)
                      for repeats in range(self.config['fpn_cell_repeats'])]

    def call(self, features, training):
        for cell in self.cells:
            cell_features = cell(features, training)
            min_level = self.config['min_level']
            max_level = self.config['max_level']

            features = []
            for level in range(min_level, max_level + 1):
                for fnode_arg, fnode in \
                        enumerate(reversed(self.fpn_config['nodes'])):
                    if fnode['feature_level'] == level:
                        features.append(cell_features[-1 - fnode_arg])
                        break
        return features


class FPNCell(Layer):
    """A single FPN cell."""
    def __init__(self, config, name='fpn_cell'):
        super().__init__(name=name)
        self.config = config
        self.fpn_config = get_fpn_configuration(config['fpn_name'],
                                                config['min_level'],
                                                config['max_level'],
                                                config['fpn_weight_method']
                                                )
        self.fnodes = []
        for fnode_cfg_arg, fnode_cfg in enumerate(self.fpn_config['nodes']):
            fnode = FNode(
                fnode_cfg['feature_level'] - self.config['min_level'],
                fnode_cfg['inputs_offsets'],
                config['fpn_num_filters'],
                config['apply_bn_for_resampling'],
                config['conv_after_downsample'],
                config['conv_bn_act_pattern'],
                config['separable_conv'],
                config['act_type'],
                weight_method=self.config['fpn_weight_method'],
                data_format=self.config['data_format'],
                name='fnode%d' % fnode_cfg_arg)
            self.fnodes.append(fnode)

    def call(self, features, training):
        for fnode in self.fnodes:
            features = fnode(features, training)
        return features


def add_n(nodes):
    """A customized add_n to add up a list of tensors."""
    with tf.name_scope('add_n'):
        new_node = nodes[0]
        for n in nodes[1:]:
            new_node = new_node + n
        return new_node


class FNode(Layer):
    """A keras layer implementing BiFPN node."""
    def __init__(self,
                 feature_level,
                 inputs_offsets,
                 fpn_num_filters,
                 apply_bn_for_resampling,
                 conv_after_downsample,
                 conv_bn_act_pattern,
                 separable_conv,
                 act_type,
                 weight_method,
                 data_format,
                 name='fnode'):
        super().__init__(name=name)
        self.feature_level = feature_level
        self.inputs_offsets = inputs_offsets
        self.fpn_num_filters = fpn_num_filters
        self.apply_bn_for_resampling = apply_bn_for_resampling
        self.conv_after_downsample = conv_after_downsample
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.separable_conv = separable_conv
        self.act_type = act_type
        self.weight_method = weight_method
        self.data_format = data_format
        self.resample_layers = []
        self.vars = []

    def fuse_features(self, nodes):
        """Fuse features from different resolutions and return a weighted sum.
        # Arguments
            nodes: a list of tensorflow features at different levels.

        # Returns
            A tensor denoting the fused features.
        """
        dtype = nodes[0].dtype

        if self.weight_method == 'attn':
            edge_weights = [tf.cast(var, dtype=dtype) for var in self.vars]
            normalized_weights = tf.nn.softmax(tf.stack(edge_weights))
            nodes = tf.stack(nodes, axis=-1)
            new_node = tf.reduce_sum(nodes * normalized_weights, -1)
        elif self.weight_method == 'fastattn':
            edge_weights = [tf.nn.relu(tf.cast(var, dtype=dtype))
                            for var in self.vars]
            weight_sum = add_n(edge_weights)
            nodes = [nodes[i] * edge_weights[i] / (weight_sum + 0.0001)
                     for i in range(len(nodes))]
            new_node = add_n(nodes)
        else:
            raise ValueError('unknown weight_method %s' % self.weight_method)

        return new_node

    def _add_scalar_multidimensional_weights(self, initializer, shape=None):
        for input_offset_arg, _ in enumerate(self.inputs_offsets):
            name = 'WSM' + ('' if input_offset_arg == 0
                            else '_' + str(input_offset_arg))
            self.vars.append(self.add_weight(name=name,
                                             initializer=initializer,
                                             shape=shape))

    def build(self, features_shape):
        for input_offset_arg, input_offset in enumerate(self.inputs_offsets):
            name = 'resample_{}_{}_{}'.format(input_offset_arg,
                                              input_offset,
                                              len(features_shape))
            self.resample_layers.append(ResampleFeatureMap(
                self.feature_level,
                self.fpn_num_filters,
                self.apply_bn_for_resampling,
                self.conv_after_downsample,
                self.data_format,
                name=name
            ))

        if self.weight_method == 'attn':
            self._add_scalar_multidimensional_weights('ones')
        elif self.weight_method == 'fastattn':
            self._add_scalar_multidimensional_weights('ones')
        self.op_after_combine = ConvolutionAfterFusion(
            self.conv_bn_act_pattern,
            self.separable_conv,
            self.fpn_num_filters,
            self.act_type,
            self.data_format,
            name='op_after_combine{}'.format(len(features_shape))
        )
        self.built = True
        super().build(features_shape)

    def call(self, features, training):
        nodes = []
        for input_offset_arg, input_offset in enumerate(self.inputs_offsets):
            input_node = features[input_offset]
            input_node = self.resample_layers[input_offset_arg](input_node,
                                                                training,
                                                                features)
            nodes.append(input_node)
        new_node = self.fuse_features(nodes)
        new_node = self.op_after_combine(new_node)
        return features + [new_node]


class ConvolutionAfterFusion(Layer):
    """Operation after combining input features during feature fusion."""
    def __init__(self,
                 conv_bn_act_pattern,
                 separable_conv,
                 fpn_num_filters,
                 act_type,
                 data_format,
                 name='op_after_combine'):
        super().__init__(name=name)
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.separable_conv = separable_conv
        self.fpn_num_filters = fpn_num_filters
        self.act_type = act_type
        self.data_format = data_format

        if self.separable_conv:
            conv2d_layer = functools.partial(tf.keras.layers.SeparableConv2D,
                                             depth_multiplier=1)
        else:
            conv2d_layer = tf.keras.layers.Conv2D

        self.conv_op = conv2d_layer(
            filters=fpn_num_filters,
            kernel_size=(3, 3),
            padding='same',
            use_bias=not self.conv_bn_act_pattern,
            data_format=self.data_format,
            name='conv'
        )
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, new_node, training):
        if not self.conv_bn_act_pattern:
            new_node = tf.nn.swish(new_node)
        new_node = self.conv_op(new_node)
        new_node = self.bn(new_node, training=training)
        if self.conv_bn_act_pattern:
            new_node = tf.nn.swish(new_node)
        return new_node
