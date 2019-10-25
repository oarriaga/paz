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
        input_topics: List of strings indicating the keys of data
            dictionary (data topics).
        output_topics: List of strings indicating the keys of data
            dictionary (data topics).
        as_dict: Boolean. If ``True`` output will be a dictionary
            of form {'inputs':list_of_input_arrays,
                     'outputs': list_of_output_arrays}
            If ``False'', output will be of the form
                list_of_input_arrays + list_of_output_arrays
    """
    def __init__(self, input_topics, label_topics):
        self.input_topics, self.label_topics = input_topics, label_topics
        super(OutputSelector, self).__init__()

    def call(self, kwargs):
        inputs, labels = {}, {}
        for topic in self.input_topics:
            inputs[topic] = kwargs[topic]
        for topic in self.label_topics:
            labels[topic] = kwargs[topic]
        return {'inputs': inputs, 'labels': labels}
