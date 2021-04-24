import tensorflow as tf


def box_refinement_graph(box, prior_box):
    """Compute refinement needed to transform box to prior_box

    # Arguments:
        box: [N, (y_min, x_min, y_max, x_max)]
        prior_box: Ground-truth box [N, (y_min, x_min, y_max, x_max)]
    """
    box = tf.cast(box, tf.float32)
    prior_box = tf.cast(prior_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    prior_height = prior_box[:, 2] - prior_box[:, 0]
    prior_width = prior_box[:, 3] - prior_box[:, 1]
    prior_center_y = prior_box[:, 0] + 0.5 * prior_height
    prior_center_x = prior_box[:, 1] + 0.5 * prior_width

    dY = (prior_center_y - center_y) / height
    dX = (prior_center_x - center_x) / width
    dH = tf.math.log(prior_height / height)
    dW = tf.math.log(prior_width / width)

    result = tf.stack([dY, dX, dH, dW], axis=1)
    return result


def batch_slice(inputs, graph_function, batch_size, names=None):
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
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_function(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
       are padded with zeros. This removes zero boxes.

    # Arguments:
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.

    # Arguments:
        boxes: [N, (y_min, x_min, y_max, x_max)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    H = boxes[:, 2] - boxes[:, 0]
    W = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * H
    center_x = boxes[:, 1] + 0.5 * W

    center_y += deltas[:, 0] * H
    center_x += deltas[:, 1] * W
    H *= tf.exp(deltas[:, 2])
    W *= tf.exp(deltas[:, 3])

    y_min = center_y - 0.5 * H
    x_min = center_x - 0.5 * W
    y_max = y_min + H
    x_max = x_min + W
    result = tf.stack([y_min, x_min, y_max, x_max], axis=1,
                      name='apply_box_deltas_out')
    return result


def clip_boxes_graph(boxes, window):
    """Clips boxes for given window size

    # Arguments:
        boxes: [N, (y_min, x_min, y_max, x_max)]
        window: [4] in the form y_min, x_min, y_max, x_max
    """
    window_y_min, window_x_min, window_y_max, window_x_max = \
        tf.split(window, 4)
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
    y_min = tf.maximum(tf.minimum(y_min, window_y_max), window_y_min)
    x_min = tf.maximum(tf.minimum(x_min, window_x_max), window_x_min)
    y_max = tf.maximum(tf.minimum(y_max, window_y_max), window_y_min)
    x_max = tf.maximum(tf.minimum(x_max, window_x_max), window_x_min)

    clipped = tf.concat([y_min, x_min, y_max, x_max], axis=1,
                        name='clipped_boxes')
    clipped.set_shape((clipped.shape[0], 4))
    return clipped
