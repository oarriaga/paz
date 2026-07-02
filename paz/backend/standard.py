from functools import wraps, partial
from collections import namedtuple
import argparse

import numpy as np
import jax
import jax.numpy as jp


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


def snapshot_variables(model):
    if model is None:
        return None
    train = [variable.value for variable in model.trainable_variables]
    nontrain = [variable.value for variable in model.non_trainable_variables]
    return train, nontrain


def model_device(model):
    return next(iter(model.weights[0].value.devices()))


def place_on_model_device(tree, model):
    device = model_device(model)
    return jax.tree.map(lambda leaf: jax.device_put(leaf, device), tree)


def call_stateless(model, variables, *inputs):
    # Run a Keras model from variables passed as inputs so jax.jit treats the
    # weights as arguments instead of constant-folding large tables into the
    # compiled executable (which would exhaust host memory).
    output, _ = model.stateless_call(variables[0], variables[1], *inputs)
    return output


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
