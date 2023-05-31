import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


def compute_ROI_level(boxes, image_shape):
    """Used by PyramidROIAlign to compute ROI levels and scaled area of the images.

    # Arguments:
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array
        image_shape: shape of image
    """
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=2)
    H = y_max - y_min
    W = x_max - x_min
    scaled_area = compute_scaled_area(H, W, image_shape)
    roi_level = compute_max_ROI_level(scaled_area)

    return tf.squeeze(roi_level, 2)


def compute_max_ROI_level(scaled_area):
    """Used by compute_ROI_level to calculate the ROI level at each feature map.

    # Arguments:
        scaled_area: area of image scaled
    """
    roi_level = tf.experimental.numpy.log2(scaled_area)
    cast_roi_level = tf.cast(tf.round(roi_level), tf.int32)
    max_roi_level = tf.maximum(2, 4 + cast_roi_level)
    roi_level = tf.minimum(5, max_roi_level)
    return roi_level


def compute_scaled_area(H, W, image_shape):
    """Used by compute_ROI_level to obtain scaled area of the image.

    # Arguments:
        H: height of image
        W: width of iamge
        image_area: area of image
    """
    image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
    image_area_scaled = 224.0 / tf.sqrt(image_area)
    squared_area = tf.sqrt(H * W)
    scaled_area = squared_area / image_area_scaled

    return scaled_area


def apply_ROI_pooling(roi_level, boxes, feature_maps, pool_shape):
    """Used by PyramidROIAlign to pool all the feature maps and bounding_box at each level.

    # Arguments:
        roi_level
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array
        features_maps: List of feature maps from different levels
                      of the pyramid. Each is [batch, height, width, channels]
        pool_shape: [pool_height, pool_width] of the output pooled regions
    """
    pooled, box_to_level = [], []
    for index, level in enumerate(range(2, 6)):
        level_index = tf.where(tf.equal(roi_level, level))
        level_boxes = tf.gather_nd(boxes, level_index)
        box_indices = tf.cast(level_index[:, 0], tf.int32)
        box_to_level.append(level_index)

        level_boxes = tf.stop_gradient(level_boxes)
        box_indices = tf.stop_gradient(box_indices)
        crop_feature_maps = tf.image.crop_and_resize(feature_maps[index], level_boxes,
                                                     box_indices, pool_shape, method='bilinear')
        pooled.append(crop_feature_maps)
    pooled = tf.concat(pooled, axis=0)
    box_to_level = tf.concat(box_to_level, axis=0)

    return pooled, box_to_level


def rearrange_pooled_features(pooled, box_to_level, boxes):
    """Used by PyramidROIAlign to reshape the pooled features.

    # Arguments:
        pooled: [batch, num_boxes, pool_height, pool_width, channels].
                The width and height are those specific in the pool_shape in the layer
                constructor.
        box_to_level: bounding_box at each level
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array
    """
    sorting_tensor = (box_to_level[:, 0] * 100000) + box_to_level[:, 1]
    top_k_indices = tf.nn.top_k(sorting_tensor, k=tf.shape(
        box_to_level)[0]).indices[::-1]
    top_k_indices = tf.gather(box_to_level[:, 2], top_k_indices)
    pooled = tf.gather(pooled, top_k_indices)
    shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)

    return tf.reshape(pooled, shape)


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
