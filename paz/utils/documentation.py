def docstring(original):
    """Doctors (documents) `target` `Callable` with `original` docstring.

    # Arguments:
        original: Object with documentation string.

    # Returns
        Function that replaces `target` docstring with `original` docstring.
    """
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target
    return wrapper
