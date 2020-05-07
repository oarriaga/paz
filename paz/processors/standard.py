import numpy as np

from ..abstract import Processor
from ..backend.boxes import to_one_hot


class ControlFlow(Processor):
    def __init__(self, processor, intro_indices=[0], outro_indices=[0]):
        self.processor = processor
        if not isinstance(intro_indices, list):
            raise ValueError('``intro_indices`` must be a list')
        if not isinstance(outro_indices, list):
            raise ValueError('``outro_indices`` must be a list')
        self.intro_indices = intro_indices
        self.outro_indices = outro_indices
        name = '-'.join([self.__class__.__name__, self.processor.name])
        super(ControlFlow, self).__init__(name)

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
            selections = [selections]
        args = self._insert(args, self.outro_indices, selections)
        return tuple(args)


class UnpackDictionary(Processor):
    def __init__(self, order):
        if not isinstance(order, list):
            raise ValueError('``order`` must be a list')
        self.order = order
        super(UnpackDictionary, self).__init__()

    def call(self, kwargs):
        args = tuple([kwargs[name] for name in self.order])
        return args


class ExtendInputs(Processor):
    def __init__(self, processor):
        self.processor = processor
        name = '-'.join([self.__class__.__name__, self.processor.name])
        super(ExtendInputs, self).__init__(name)

    def call(self, X, *args):
        return self.processor(X), *args


class OutputWrapper(Processor):
    def __init__(self, inputs_info, labels_info):
        if not isinstance(inputs_info, dict):
            raise ValueError('``inputs_info`` must be a dictionary')
        self.inputs_info = inputs_info
        if not isinstance(labels_info, dict):
            raise ValueError('``inputs_info`` must be a dictionary')
        self.labels_info = labels_info
        self.inputs_name_to_shape = self._extract_name_to_shape(inputs_info)
        self.labels_name_to_shape = self._extract_name_to_shape(labels_info)
        super(OutputWrapper, self).__init__()

    def _extract_name_to_shape(self, info):
        name_to_shape = list(info.values())
        if len(name_to_shape) != 1:
            raise ValueError('``values`` of ``info`` must be a single dict')
        return name_to_shape[0]

    def _wrap(self, args, info):
        wrap = {}
        for arg, name_to_shape in info.items():
            name = list(name_to_shape.keys())[0]
            wrap[name] = args[arg]
        return wrap

    def call(self, *args):
        inputs = self._wrap(args, self.inputs_info)
        labels = self._wrap(args, self.labels_info)
        return {'inputs': inputs, 'labels': labels}


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
    def __init__(self):
        super(Copy, self).__init__()

    def call(self, X):
        return X.copy()


class Lambda(object):
    """Applies a lambda function as a processor transformation.
    # Arguments
        function: Function.
        parameters: Dictionary.
        topic: String
    """

    def __init__(self, function):
        self.function = function

    def __call__(self, X):
        return self.function(X)
