from ..core import SequentialProcessor
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


class PrintTopics(Processor):
    def __init__(self, topics):
        self.topics = topics
        super(PrintTopics, self).__init__()

    def call(self, kwargs):
        for topic in self.topics:
            print(topic, kwargs[topic])
        return kwargs


class Predict(Processor):
    def __init__(self, model, input_topic, label_topic='predictions',
                 processors=None):

        super(Predict, self).__init__()
        self.model = model
        self.processors = processors
        if self.processors is not None:
            self.process = SequentialProcessor(processors)
        self.input_topic = input_topic
        self.label_topic = label_topic

    def call(self, kwargs):
        input_topic = kwargs[self.input_topic]
        if self.processors is not None:
            processing_kwargs = {self.input_topic: input_topic}
            input_topic = self.process(processing_kwargs)[self.input_topic]
        label_topic = self.model.predict(input_topic)
        kwargs[self.label_topic] = label_topic
        return kwargs


class SelectElement(Processor):
    def __init__(self, topic, argument):
        super(SelectElement, self).__init__()
        self.topic = topic
        self.argument = argument

    def call(self, kwargs):
        kwargs[self.topic] = kwargs[self.topic][self.argument]
        return kwargs


class ToLabel(Processor):
    def __init__(self, class_names, topic='predictions'):
        super(ToLabel, self).__init__()
        self.class_names = class_names
        self.topic = topic

    def call(self, kwargs):
        kwargs[self.topic] = self.class_names[np.argmax(kwargs[self.topic])]
        return kwargs


class ExpandDims(Processor):
    """Wrap around numpy `expand_dims` due to common use before model predict.
    # Arguments
        expand_dims: Int or list of Ints.
        topic: String.
    """
    def __init__(self, axis, topic):
        super(ExpandDims, self).__init__()
        self.axis, self.topic = axis, topic

    def call(self, kwargs):
        kwargs[self.topic] = np.expand_dims(kwargs[self.topic], self.axis)
        return kwargs


class Squeeze(Processor):
    """Wrap around numpy `squeeze` due to common use before model predict.
    # Arguments
        expand_dims: Int or list of Ints.
        topic: String.
    """
    def __init__(self, axis, topic):
        super(Squeeze, self).__init__()
        self.axis, self.topic = axis, topic

    def call(self, kwargs):
        kwargs[self.topic] = np.squeeze(kwargs[self.topic], axis=self.axis)
        return kwargs


class Copy(Processor):
    """Copy values from ``input_topic`` to a new ``label_topic``
    # Arguments
        input_topic: String. Topic to copy from.
        label_topic: String. Topic to copy to.
    """
    def __init__(self, input_topic, label_topic):
        super(Copy, self).__init__()
        self.input_topic, self.label_topic = input_topic, label_topic

    def call(self, kwargs):
        kwargs[self.label_topic] = kwargs[self.input_topic].copy()
        return kwargs


class Lambda(object):
    """Applies a lambda function as a processor transformation.
    # Arguments
        function: Function.
        parameters: Dictionary.
        topic: String
    """

    def __init__(self, function, parameters, topic):
        self.function = function
        self.parameters = parameters
        self.topic = topic

    def __call__(self, kwargs):
        data = self.function(kwargs[self.topic], **self.parameters)
        kwargs[self.topic] = data
        return kwargs
