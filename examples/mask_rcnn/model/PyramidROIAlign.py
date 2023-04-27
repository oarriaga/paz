import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from mask_rcnn.layer_utils import compute_ROI_level, apply_ROI_pooling, rearrange_pooled_features


class PyramidROIAlign(Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    # Arguments:
        pool_shape: [pool_height, pool_width] of the output pooled regions
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array.
        image_shape: shape of image
        feature_maps: List of feature maps from different levels
                      of the pyramid. Each is [batch, height, width, channels]

    # Returns:
        Pooled regions: [batch, num_boxes, pool_height, pool_width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super().__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        boxes, image_shape = inputs[0], inputs[1]
        feature_maps = inputs[2:]

        roi_level = compute_ROI_level(boxes, image_shape)
        pooled, box_to_level = apply_ROI_pooling(roi_level, boxes,
                                                 feature_maps, self.pool_shape)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)
        pooled = rearrange_pooled_features(pooled, box_to_level, boxes)
        return pooled
