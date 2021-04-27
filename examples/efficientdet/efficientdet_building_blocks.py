import tensorflow as tf


class ResampleFeatureMap(tf.keras.layers.Layer):
    """Resample feature maps for downsampling or upsampling to create coarser or finer additional feature maps."""

    def __init__(self,
                 feat_level,
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
        self.feat_level = feat_level
        self.conv_after_downsample = conv_after_downsample
        self.pooling_type = pooling_type or 'max'
        self.upsampling_type = upsampling_type or 'nearest'
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)
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
        raise ValueError('Unsupported pooling type {}.'.format(self.pooling_type))

    def _upsample2d(self, inputs, target_height, target_width):
        return tf.cast(
            tf.compat.v1.image.resize_nearest_neighbor(
                tf.cast(inputs, tf.float32), [target_height, target_width]),
            inputs.dtype)

    def _maybe_apply_1x1(self, feat, training, num_channels):
        """Apply 1x1 conv to change layer width if necessary."""
        if num_channels != self.target_num_channels:
            feat = self.conv2d(feat)
            if self.apply_bn:
                feat = self.bn(feat, training=training)
        return feat

    def call(self, feat, training, all_feats):
        hwc_idx = (2, 3, 1) if self.data_format == 'channels_first' else (1, 2, 3)
        height, width, num_channels = [feat.shape.as_list()[i] for i in hwc_idx]
        if all_feats:
            target_feat_shape = all_feats[self.feat_level].shape.as_list()
            target_height, target_width, _ = [target_feat_shape[i] for i in hwc_idx]
        else:
            # Default to downsampling if all_feats is empty.
            target_height, target_width = (height + 1) // 2, (width + 1) // 2

        # If conv_after_downsample is True, when downsampling, apply 1x1 after
        # downsampling for efficiency.
        if height > target_height and width > target_width:
            if not self.conv_after_downsample:
                feat = self._maybe_apply_1x1(feat, training, num_channels)
            feat = self._pool2d(feat, height, width, target_height, target_width)
            if self.conv_after_downsample:
                feat = self._maybe_apply_1x1(feat, training, num_channels)
        elif height <= target_height and width <= target_width:
            feat = self._maybe_apply_1x1(feat, training, num_channels)
            if height < target_height or width < target_width:
                feat = self._upsample2d(feat, target_height, target_width)
        else:
            raise ValueError('Incompatible Resampling : feat shape {}x{} \
            target_shape: {}x{}'.format(height, width, target_height, target_width))

        return feat


class FPNCells(tf.keras.layers.Layer):

    pass
