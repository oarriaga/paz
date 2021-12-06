import cv2
import numpy as np


def wrap_as_dictionary(keys, values):
    """ Wrap values with respective keys into a dictionary.

    # Arguments
        keys: List of strings.
        Values: List.

    # Returns
        output: Dictionary.
    """
    output = dict(zip(keys, values))
    return output


def merge_dictionaries(dicts):
    """ Merge multiple dictionaries.

    # Arguments
        dicts: List of dictionaries.

    # Returns
        result: Dictionary.
    """
    result = {}
    for dict in dicts:
        result.update(dict)
    return result


def resize_image_with_linear_interpolation(image, size):
    """Resize image using nearest neighbors interpolation.

    # Arguments
        image: Numpy array.
        size: List of two ints.

    # Returns
        Numpy array.
    """
    if(type(image) != np.ndarray):
        raise ValueError(
            'Recieved Image is not of type numpy array', type(image))
    else:
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def transpose_array(array):
    """Resize image using nearest neighbors interpolation.

    # Arguments
        image: Numpy array.
        size: List of two ints.

    # Returns
        Numpy array.
    """
    if(type(array) != np.ndarray):
        raise ValueError(
            'Recieved Input is not of type numpy array', type(array))
    else:
        return array.T
