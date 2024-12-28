import jax.numpy as jp


def pad(class_args, size, value=-1):
    """Pads class_args with given value.

    # Arguments
        class_args: Array `(num_boxes, 4)`.

    # Returns
        Padded class_args with shape `(size, 4)`.
    """
    num_classes = len(class_args)
    if num_classes > size:
        raise ValueError(f"Samples ({num_classes}) exceeds pad ({size}).")
    padding = (0, size - num_classes)
    return jp.pad(class_args, padding, "constant", constant_values=value)


def pad_data(class_args, size, value=-1):
    """Pads list of class_args with a given.

    # Arguments
        class_args: List of size `(num_samples)` containing list of class args.

    # Returns
        class_args with shape `(num_samples, size, 4)`
    """
    padded_elements = []
    for sample in class_args:
        padded_element = pad(jp.array(sample), size, value)
        padded_elements.append(padded_element)
    return jp.array(padded_elements)
