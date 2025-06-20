import jax
import jax.numpy as jp


def pad_batch(ragged_class_args, size, value=-1):
    """Pads class_args with given value."""
    return jp.array([pad(jp.array(x), size, value) for x in ragged_class_args])


def pad(class_args, size, value=-1):
    class_args = class_args[:size]
    padding = (0, size - len(class_args))
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


def to_one_hot(class_args, num_classes):
    """Transform from class index to one-hot encoded vector."""
    return jax.nn.one_hot(class_args, num_classes)


def join(list_of_class_args):
    return jp.concatenate(list_of_class_args, axis=0)
