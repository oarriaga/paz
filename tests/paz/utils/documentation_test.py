from paz.utils import docstring


def documented_function():
    """This is a dummy function
    """
    return None


@docstring(documented_function)
def undocumented_function():
    return None


def test_docstring():
    assert documented_function.__doc__ == undocumented_function.__doc__
