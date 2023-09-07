import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


def compute_ROI_level(boxes, image_shape):
    """Computes ROI levels and scaled area of the images for use in
    PyramidROIAlign.

    # Arguments:
        boxes: A tensor of shape [batch, num_boxes, (x1, y1, x2, y2)]
               representing the bounding boxes in normalized coordinates. It
               may be padded with zeros if there are not enough boxes to fill
               the array.
        image_shape: A tensor or tuple of integers representing the shape of
                     the image [height, width].

    # Returns:
        ROI_level: A tensor of shape [batch, num_boxes] containing the computed
                   ROI levels.

    """
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=2)
    H = y_max - y_min
    W = x_max - x_min
    scaled_area = compute_scaled_area(H, W, image_shape)
    ROI_level = compute_max_ROI_level(scaled_area)
    return tf.squeeze(ROI_level, 2)


def compute_max_ROI_level(scaled_area):
    """Calculates the maximum ROI level at each feature map based on the
    scaled area.

    # Arguments:
        scaled_area: A tensor representing the scaled area of the image.

    # Returns:
        ROI_level: A tensor representing the maximum ROI level at each feature
                   map.

    """
    ROI_level = tf.experimental.numpy.log2(scaled_area)
    cast_ROI_level = tf.cast(tf.round(ROI_level), tf.int32)
    max_ROI_level = tf.maximum(2, 4 + cast_ROI_level)
    ROI_level = tf.minimum(5, max_ROI_level)
    return ROI_level


def compute_scaled_area(H, W, image_shape):
    """Calculates the scaled area of the image for use in compute_ROI_level.

    # Arguments:
        H: The height of the image.
        W: The width of the image.
        image_shape: A tensor or tuple of integers representing the shape of
                     the image [width, height].

    # Returns:
        scaled_area: A tensor representing the scaled area of the image.
    """
    image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
    image_area_scaled = 224.0 / tf.sqrt(image_area)
    squared_area = tf.sqrt(H * W)
    scaled_area = squared_area / image_area_scaled
    return scaled_area


def apply_ROI_pooling(ROI_level, boxes, feature_maps, pool_shape):
    """Pools feature maps and bounding boxes at each level based on ROI levels.

    # Arguments:
        ROI_level: A 1-D tensor of shape [N] representing the ROI levels
                   Each element corresponds to the level of the pyramid to
                   which the ROI belongs.
        boxes: A tensor of shape [batch, num_boxes, (x1, y1, x2, y2)]
               representing the bounding boxes in normalized coordinates. It
               may be padded with zeros if there are not enough boxes to fill
               the array.
        feature_maps: A list of feature maps from different levels of the
                      pyramid. Each feature map has shape
                      [batch, height, width, channels].
        pool_shape: A tuple or list of two integers representing the desired
                    output pooled region shape [pool_height, pool_width].

    # Returns:
        pooled: A tensor of shape.
                [num_pooled_regions, pool_height, pool_width, channels]
                representing the pooled regions from different levels of the
                pyramid.
        box_to_level: A tensor of shape [num_boxes] containing the level index
                      corresponding to each box in the pooled tensor.

    """
    pooled, box_to_level = [], []
    for index, level in enumerate(range(2, 6)):
        level_index = tf.where(tf.equal(ROI_level, level))
        level_boxes = tf.gather_nd(boxes, level_index)
        box_indices = tf.cast(level_index[:, 0], tf.int32)
        box_to_level.append(level_index)

        level_boxes = tf.stop_gradient(level_boxes)
        box_indices = tf.stop_gradient(box_indices)
        crop_feature_maps = tf.image.crop_and_resize(feature_maps[index],
                                                     level_boxes,
                                                     box_indices,
                                                     pool_shape,
                                                     method='bilinear')
        pooled.append(crop_feature_maps)
    pooled = tf.concat(pooled, axis=0)
    box_to_level = tf.concat(box_to_level, axis=0)
    return pooled, box_to_level


def rearrange_pooled_features(pooled, box_to_level, boxes):
    """Reshapes the pooled features for use in PyramidROIAlign.

    This function is used by the `PyramidROIAlign` layer to reshape the pooled
    features based on the provided box-to-level mapping and bounding boxes.

    # Arguments:
        pooled: A tensor of shape
                [batch, num_boxes, pool_height, pool_width, channels]
                The width and height are specific to the pool_shape specified
                in the layer constructor.
        box_to_level: A tensor representing the bounding boxes at each level
        boxes: A tensor of shape [batch, num_boxes, (x1, y1, x2, y2)] in
               normalized coordinates. It may be padded with zeros if there
               are not enough boxes to fill the array.

    # Returns:
        pooled_boxes: A tensor of shape
                      [batch, num_boxes, pool_height, pool_width, channels].

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

    This layer performs ROI pooling on multiple levels of the feature pyramid.
    Given the input boxes, image shape, and feature maps, it pools the regions
    of interest from the feature maps based on the provided pool shape.

    # Arguments:
        pool_shape: A tuple of [pool_height, pool_width] specifying the output
                    pooled region shape.
        boxes: A tensor of shape [batch, num_boxes, (x1, y1, x2, y2)] in
               normalized coordinates. It may be padded with zeros if there are
               not enough boxes to fill the array.
        image_shape: A tensor or tuple of integers representing the shape of
                     the image [width, height].
        feature_maps: A list of feature maps from different levels of the
                      pyramid. Each feature map has a shape of [batch, height,
                      width, channels].

    # Returns:
        pooled_regions: A tensor of shape
                        [batch, num_boxes, pool_height, pool_width, channels]
                        The width and height are specific to the pool_shape
                        specified in the layer constructor.
    """
    def __init__(self, pool_shape, **kwargs):
        super().__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        boxes, image_shape = inputs[0], inputs[1]
        feature_maps = inputs[2:]

        ROI_level = compute_ROI_level(boxes, image_shape)
        pooled, box_to_level = apply_ROI_pooling(ROI_level, boxes,
                                                 feature_maps, self.pool_shape)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)
        pooled = rearrange_pooled_features(pooled, box_to_level, boxes)
        return pooled
