import tensorflow as tf
import numpy as np


def slice_batch(inputs, constants, function, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results.

    This function takes a list of tensors `inputs` and splits them into
    `batch_size` slices. Each slice is then passed to a copy of the provided
    computation graph `function`, along with the corresponding slice of
    constants `constants`. The outputs from each slice are combined
    and returned as a list of tensors.

    # Arguments:
        inputs: A list of tensors. All tensors must have the same first
                dimension length.
        constants: A list of tensors representing the constants to be passed
                   along with each input slice.
        function: A function that takes the input slices and constants as
                  arguments and returns a tensor.
        batch_size: The number of slices to divide the data into.
        names: Optional. If provided, assigns names to the resulting tensors.

    # Returns:
        results: A list of tensors containing the outputs from each slice. The
                 tensors are stacked along the first dimension.
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
    """ Applies the given deltas to the given boxes.

    Arguments:
        boxes: A tensor of shape [N, 4] representing the boxes to update.
               Each box is represented as (y_min, x_min, y_max, x_max).
        deltas: A tensor of shape [N, 4] representing the refinements to apply.
                Each delta is represented as (dy, dx, log(dh), log(dw)).

    Returns:
        A tensor of shape [N, 4] representing the updated boxes.
        Each box is represented as (y_min, x_min, y_max, x_max).
    """
    boxes = tf.cast(boxes, tf.float32)
    H = boxes[:, 2] - boxes[:, 0]
    W = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + (0.5 * H)
    center_x = boxes[:, 1] + (0.5 * W)

    center_y = center_y + (deltas[:, 0] * H)
    center_x = center_x + (deltas[:, 1] * W)
    W = W * tf.exp(deltas[:, 2])
    H = H * tf.exp(deltas[:, 3])

    x_min = center_x - (0.5 * W)
    y_min = center_y - (0.5 * H)
    y_max = y_min + H
    x_max = x_min + W
    result = tf.stack([y_min, x_min, y_max, x_max], axis=1,
                      name='apply_box_deltas_out')
    return result


def clip_boxes(boxes, window):
    """Clips the boxes to fit within a given window size.

    Arguments:
        boxes: A tensor of shape [N, 4] representing the boxes to be clipped.
               Each box is represented as (y_min, x_min, y_max, x_max).
        window: A tensor of shape [4] representing the window size.
                The window is defined as (y_min, x_min, y_max, x_max).

    Returns:
        A tensor of shape [N, 4] representing the clipped boxes.
        Each box is represented as (y_min, x_min, y_max, x_max).
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
