from paz.abstract import Processor
from backend_standard import wrap_dictionary, merge_dictionaries
from paz.backend.boxes import to_one_hot


class WraptoDictionary(Processor):
    """ Wrap the input values to a dictionary with already provided key
    values """

    def __init__(self, keys):
        super(WraptoDictionary, self).__init__()
        if not isinstance(keys, list):
            keys = list(keys)
        self.keys = keys

    def call(self, values):
        if not isinstance(values, list):
            values = list(values)
        return wrap_dictionary(self.keys, values)


class MergeDictionaries(Processor):
    """ Merge two dictionaries into one"""

    def __init__(self):
        super(MergeDictionaries, self).__init__()

    def call(self, dicts):
        return merge_dictionaries(dicts)


class ToOneHot(Processor):
    """Extract Hand mask."""

    def __init__(self, num_classes=2):
        super(ToOneHot, self).__init__()
        self.num_classes = num_classes

    def call(self, class_indices):
        return to_one_hot(class_indices, self.num_classes)
