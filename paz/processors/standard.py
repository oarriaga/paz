from ..core import Processor
from ..core import ops
import numpy as np


class ToOneHotVector(Processor):
    """Transform from class index to a one-hot encoded vector.
    # Arguments
        num_classes: Integer. Total number of classes.
        topic: String. Currently valid topics: `boxes`
    """
    def __init__(self, num_classes, topic='boxes'):
        self.num_classes = num_classes
        self.topic = topic
        super(ToOneHotVector, self).__init__()

    def call(self, kwargs):
        if self.topic == 'boxes':
            boxes = kwargs[self.topic]
            class_indices = boxes[:, 4].astype('int')
            one_hot_vectors = ops.to_one_hot(class_indices, self.num_classes)
            one_hot_vectors = one_hot_vectors.reshape(-1, self.num_classes)
            boxes = np.hstack([boxes[:, :4], one_hot_vectors.astype('float')])
            kwargs[self.topic] = boxes
        return kwargs


class OutputSelector(Processor):
    """Selects data types (topics) that will be outputted.
    #Arguments
        topics: List of strings. The topics will be outputted in the
            same order as given.
    """
    def __init__(self, topics):
        self.topics = topics
        super(OutputSelector, self).__init__()

    def call(self, kwargs):
        return [kwargs[topic] for topic in self.topics]


class InputSelector(Processor):
    """Selects data types (topics) that will be outputted.
    #Arguments
        topics: List of strings. The topics will be outputted in the
            same order as given.
    """
    def __init__(self, topics):
        self.topics = topics
        super(InputSelector, self).__init__()

    def call(self, **kwargs):
        return [kwargs[topic] for topic in self.topics]
