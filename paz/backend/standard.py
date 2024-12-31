from functools import wraps


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


def cast(x, dtype):
    """Casts array to different type
    """
    return x.astype(dtype)
