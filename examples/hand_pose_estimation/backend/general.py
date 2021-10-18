def wrap_dictionary(keys, values):
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