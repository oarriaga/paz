from functools import wraps, partial
from collections import namedtuple
import argparse

import h5py
import numpy as np
import jax.numpy as jp


def load_weights_by_name(model, weights_path):
    """Loads legacy HDF5 weights into a model by matching layer names.

    Used when a clean model builds its layers in a different order than the
    original save, so loading by save order would misalign weights.
    """
    with h5py.File(weights_path, "r") as weights:
        for layer_name in decode_strings(weights.attrs["layer_names"]):
            group = weights[layer_name]
            weight_names = decode_strings(group.attrs.get("weight_names", []))
            if weight_names:
                arrays = [np.array(group[name]) for name in weight_names]
                model.get_layer(layer_name).set_weights(arrays)


def decode_strings(names):
    return [n.decode() if isinstance(n, bytes) else n for n in names]


def merge_dicts(a, b):
    """Merges two dictionaries

    # Arguments
        a: Dictionary.
        b: Dictionary.

    # Returns
        Dictionary with all elements and values of `a` and `b`.
    """
    return {**a, **b}


def lock(function, *args, **kwargs):
    """Same as `functools.partial` but fills arguments from right to left."""

    @wraps(function)
    def wrap(*remaining_args, **remaining_kwargs):
        combined_args = remaining_args + args
        combined_kwargs = merge_dicts(remaining_kwargs, kwargs)
        return function(*combined_args, **combined_kwargs)

    return wrap


def NamedTuple(class_name, **fields):
    return namedtuple(class_name, fields)(*fields.values())


def cast(x, dtype):
    """Casts array to different type"""
    return x.astype(dtype)


def to_numpy(x):
    return np.array(x, dtype=x.dtype)


def to_jax(x):
    return jp.array(x)


def as_numpy_array(function):
    """Decorator to convert the output of a function into a NumPy array."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        return np.array(result)

    return wrapper


def str_to_bool(value):
    if isinstance(value, bool):
        result = value
    else:
        value = value.lower()
        if value in {"true", "1", "yes", "y"}:
            result = True
        elif value in {"false", "0", "no", "n"}:
            result = False
        else:
            raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
    return result
