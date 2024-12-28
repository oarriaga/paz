import jax.numpy as jp


def split(boxes):
    return jp.split(boxes, 4, axis=1)


def to_center_form(boxes):
    """Transform from corner coordinates to center coordinates.

    # Arguments
        boxes: Numpy array with shape `(num_boxes, 4)`.

    # Returns
        Numpy array with shape `(num_boxes, 4)`.
    """
    x_min, y_min, x_max, y_max = split(boxes)
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    W = x_max - x_min
    H = y_max - y_min
    return jp.concatenate([center_x, center_y, W, H], axis=1)


def to_corner_form(boxes):
    """Transform from center coordinates to corner coordinates.

    # Arguments
        boxes: Numpy array with shape `(num_boxes, 4)`.

    # Returns
        Numpy array with shape `(num_boxes, 4)`.
    """
    center_x, center_y, W, H = split(boxes)
    x_min = center_x - (W / 2.0)
    x_max = center_x + (W / 2.0)
    y_min = center_y - (H / 2.0)
    y_max = center_y + (H / 2.0)
    return jp.concatenate([x_min, y_min, x_max, y_max], axis=1)


def pad(boxes, size, value=-1):
    """Pads boxes with given value.

    # Arguments
        boxes: Array `(num_boxes, 4)`.

    # Returns
        Padded boxes with shape `(size, 4)`.
    """
    num_boxes = len(boxes)
    if num_boxes > size:
        raise ValueError(f"Samples ({num_boxes}) exceeds pad ({size}).")
    padding = ((0, size - num_boxes), (0, 0))
    return jp.pad(boxes, padding, "constant", constant_values=value)


def pad_data(boxes, size, value=-1):
    """Pads list of boxes with a given.

    # Arguments
        boxes: List of size `(num_samples)` containing boxes as lists.

    # Returns
        boxes with shape `(num_samples, size, 4)`
    """
    padded_elements = []
    for sample in boxes:
        padded_element = pad(jp.array(sample), size, value)
        padded_elements.append(padded_element)
    return jp.array(padded_elements)
