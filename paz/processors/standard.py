import numpy as np

from ..abstract import Processor
from ..backend.boxes import to_one_hot


class ModifyFlow(Processor):
    def __init__(self, processor, intro_indices=[0], outro_indices=[0]):
        self.processor = processor
        if not isinstance(intro_indices, list):
            raise ValueError('``intro_indices`` must be a list')
        if not isinstance(outro_indices, list):
            raise ValueError('``outro_indices`` must be a list')
        self.intro_indices = intro_indices
        self.outro_indices = outro_indices
        name = '-'.join([self.__class__.__name__, self.processor.name])
        super(ModifyFlow, self).__init__(name)

    def _select(self, inputs, indices):
        return [inputs[index] for index in indices]

    def _remove(self, inputs, indices):
        return [inputs[i] for i in range(len(inputs)) if i not in indices]

    def _split(self, inputs, indices):
        return self._select(inputs, indices), self._remove(inputs, indices)

    def _insert(self, args, axes, values):
        [args.insert(axis, value) for axis, value in zip(axes, values)]
        return args

    def call(self, *args):
        selections, args = self._split(args, self.intro_indices)
        selections = self.processor(*selections)
        if not isinstance(selections, tuple):
            print(selections)
            selections = [selections]
        args = self._insert(args, self.outro_indices, selections)
        return tuple(args)


class ExtendInputs(Processor):
    def __init__(self, processor):
        self.processor = processor
        name = '-'.join([self.__class__.__name__, self.processor.name])
        super(ExtendInputs, self).__init__(name)
        print(self.processor)

    def call(self, X, *args):
        return self.processor(X), *args


class Predict(Processor):
    def __init__(self, model, preprocess=None, postprocess=None):
        super(Predict, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def call(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        y = self.model.predict(x)
        if self.postprocess is not None:
            y = self.postprocess(y)
        return y


class ToClassName(Processor):
    def __init__(self, labels):
        super(ToClassName, self).__init__()
        self.labels = labels

    def call(self, x):
        return self.labels[np.argmax(x)]


class ExpandDims(Processor):
    def __init__(self, axis):
        super(ExpandDims, self).__init__()
        self.axis = axis

    def call(self, x):
        return np.expand_dims(x, self.axis)


class BoxClassToOneHotVector(Processor):
    """Transform from class index to a one-hot encoded vector.
    # Arguments
        num_classes: Integer. Total number of classes.
        topic: String. Currently valid topics: `boxes`
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(BoxClassToOneHotVector, self).__init__()

    def call(self, boxes):
        class_indices = boxes[:, 4].astype('int')
        one_hot_vectors = to_one_hot(class_indices, self.num_classes)
        one_hot_vectors = one_hot_vectors.reshape(-1, self.num_classes)
        boxes = np.hstack([boxes[:, :4], one_hot_vectors.astype('float')])
        return boxes


class OutputWrapper(Processor):
    def __init__(self, input_names, label_names):
        self.input_names = input_names
        self.label_names = label_names
        super(OutputWrapper, self).__init__()

    def _wrap_samples(self, samples, names):
        wrap = {}
        for sample, name in zip(samples, names):
            wrap[name] = sample
        return wrap

    def call(self, inputs, labels):
        if isinstance(inputs, list):
            inputs = self._wrap_samples(inputs, self.input_names)
        else:
            if len(self.input_names) != 1:
                raise ValueError('Invalid number of ``input_names``')
            inputs = {self.input_names[0]: inputs}
        if isinstance(labels, list):
            labels = self._wrap_samples(labels, self.label_names)
            if len(self.label_names) != 1:
                raise ValueError('Invalid number of ``label_names``')
            labels = {self.label_names[0]: labels}
        return {'inputs': inputs, 'labels': labels}


class SelectElement(Processor):
    def __init__(self, topic, argument):
        super(SelectElement, self).__init__()
        self.topic = topic
        self.argument = argument

    def call(self, kwargs):
        kwargs[self.topic] = kwargs[self.topic][self.argument]
        return kwargs


class Squeeze(Processor):
    """Wrap around numpy `squeeze` due to common use before model predict.
    # Arguments
        expand_dims: Int or list of Ints.
        topic: String.
    """
    def __init__(self, axis):
        super(Squeeze, self).__init__()
        self.axis = axis

    def call(self, x):
        return np.squeeze(x, axis=self.axis)


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
