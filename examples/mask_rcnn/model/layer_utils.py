import tensorflow as tf
import numpy as np


def slice_batch(inputs, constants, function, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
       computation graph and then combines the results.

    # Arguments:
        inputs: list of tensors. All must have the same first dimension length
        graph_function: Function that returns a tensor that's part of a graph.
        batch_size: number of slices to divide the data into.
        names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    outputs = []
    for sample_arg in range(batch_size):

        input_slices = []
        for x in inputs:
            input_slice = x[sample_arg]
            input_slices.append(input_slice)
        for y in constants:
            input_slices.append(y)

        output_slice = function(*input_slices)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    results = []
    for output, name in zip(outputs, names):
        result = tf.stack(output, axis=0, name=name)
        results.append(result)

    if len(results) == 1:
        results = results[0]

    return results


def apply_box_delta(boxes, deltas):
    """Applies the given deltas to the given boxes.

    # Arguments:
        boxes: [N, (y_min, x_min, y_max, x_max)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    boxes = tf.cast(boxes, tf.float32)
    W = boxes[:, 2] - boxes[:, 0]
    H = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + (0.5 * W)
    center_x = boxes[:, 1] + (0.5 * H)

    center_y = center_y + (deltas[:, 0] * W)
    center_x = center_x + (deltas[:, 1] * H)
    W = W * tf.exp(deltas[:, 2])
    H = H * tf.exp(deltas[:, 3])

    x_min = center_x - (0.5 * W)
    y_min = center_y - (0.5 * H)
    x_max = y_min + W
    y_max = x_min + H
    result = tf.stack([x_min, y_min, x_max, y_max], axis=1)

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
