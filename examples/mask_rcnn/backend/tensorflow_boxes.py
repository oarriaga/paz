import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


def refine_bounding_box(box, prior_box):
    """Compute refinement needed to transform box to groundtruth_box.

    # Arguments:
        box: [N, (y_min, x_min, y_max, x_max)]
        prior_box: Ground-truth box [N, (y_min, x_min, y_max, x_max)]
    """
    box = tf.cast(box, tf.float32)
    prior_box = tf.cast(prior_box, tf.float32)
    x_box = box[:, 0]
    y_box = box[:, 1]

    H = box[:, 2] - x_box
    W = box[:, 3] - y_box
    center_y = x_box + (0.5 * H)
    center_x = y_box + (0.5 * W)

    prior_H = prior_box[:, 2] - prior_box[:, 0]
    prior_W = prior_box[:, 3] - prior_box[:, 1]
    prior_center_y = prior_box[:, 0] + (0.5 * prior_H)
    prior_center_x = prior_box[:, 1] + (0.5 * prior_W)

    dy = (prior_center_y - center_y) / H
    dx = (prior_center_x - center_x) / W
    dh = tf.math.log(prior_H / H)
    dw = tf.math.log(prior_W / W)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def trim_zeros(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
       are padded with zeros. This removes zero boxes.

    # Arguments:
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def apply_box_delta(boxes, deltas):
    """Applies the given deltas to the given boxes.

    # Arguments:
        boxes: [N, (y_min, x_min, y_max, x_max)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    H = boxes[:, 2] - boxes[:, 0]
    W = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + (0.5 * H)
    center_x = boxes[:, 1] + (0.5 * W)

    center_y = center_y + (deltas[:, 0] * H)
    center_x = center_x + (deltas[:, 1] * W)
    H = H * tf.exp(deltas[:, 2])
    W = W * tf.exp(deltas[:, 3])

    y_min = center_y - (0.5 * H)
    x_min = center_x - (0.5 * W)
    y_max = y_min + H
    x_max = x_min + W
    result = tf.stack([y_min, x_min, y_max, x_max], axis=1,
                      name='apply_box_deltas_out')    # TODO: remove names for tensors

    return result


def clip_boxes(boxes, window):
    """Clips boxes for given window size.

    # Arguments:
        boxes: [N, (y_min, x_min, y_max, x_max)]
        window: [4] in the form y_min, x_min, y_max, x_max
    """
    windows = tf.split(window, 4)
    window_y_min = windows[0]
    window_x_min = windows[1]
    window_y_max = windows[2]
    window_x_max = windows[3]

    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
    y_min = tf.maximum(tf.minimum(y_min, window_y_max), window_y_min)
    x_min = tf.maximum(tf.minimum(x_min, window_x_max), window_x_min)
    y_max = tf.maximum(tf.minimum(y_max, window_y_max), window_y_min)
    x_max = tf.maximum(tf.minimum(x_max, window_x_max), window_x_min)

    clipped = tf.concat([y_min, x_min, y_max, x_max], axis=1,
                        name='clipped_boxes')
    clipped.set_shape((clipped.shape[0], 4))

    return clipped


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)
