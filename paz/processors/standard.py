import numpy as np

from ..abstract import Processor
from ..backend.boxes import to_one_hot
from ..backend.standard import append_values, predict


class ControlMap(Processor):
    """Controls which inputs are passed ''processor'' and the order of its
        outputs.

    # Arguments
        processor: Function e.g. a ''paz.processor''
        intro_indices: List of Ints.
        outro_indices: List of Ints.
        keep: ''None'' or dictionary. If ``None`` control maps operates
            without explicitly retaining an input. If dict it must contain
            as keys the input args to be kept and as values where they should
            be located at the end.
    """
    def __init__(self, processor, intro_indices=[0], outro_indices=[0],
                 keep=None):
        self.processor = processor
        if not isinstance(intro_indices, list):
            raise ValueError('``intro_indices`` must be a list')
        if not isinstance(outro_indices, list):
            raise ValueError('``outro_indices`` must be a list')
        self.intro_indices = intro_indices
        self.outro_indices = outro_indices
        name = '-'.join([self.__class__.__name__, self.processor.name])
        self.keep = keep
        super(ControlMap, self).__init__(name)

    def _select(self, inputs, indices):
        return [inputs[index] for index in indices]

    def _remove(self, inputs, indices):
        return [inputs[i] for i in range(len(inputs)) if i not in indices]

    def _split(self, inputs, indices):
        return self._select(inputs, indices), self._remove(inputs, indices)

    def _insert(self, args, extra_args, indices):
        [args.insert(index, arg) for index, arg in zip(indices, extra_args)]
        return args

    def call(self, *args):
        selected_args, remaining_args = self._split(args, self.intro_indices)
        processed_args = self.processor(*selected_args)
        if not isinstance(processed_args, tuple):
            processed_args = [processed_args]
        return_args = self._insert(
            remaining_args, processed_args, self.outro_indices)

        if self.keep is not None:
            keep_intro = list(self.keep.keys())
            keep_outro = list(self.keep.values())
            keep_args = self._select(args, keep_intro)
            return_args = self._insert(return_args, keep_args, keep_outro)

        return tuple(return_args)


class ExpandDomain(ControlMap):
    """Extends number of inputs a function can take applying the identity
    function to all new/extended inputs.
    e.g. For a given function f(x) = y. If g = ExtendInputs(f), we can
    now have g(x, x1, x2, ..., xn) = y, x1, x2, ..., xn.

    # Arguments
        processor: Function e.g. any procesor in ''paz.processors''.
    """
    def __init__(self, processor):
        super(ExpandDomain, self).__init__(processor)


class CopyDomain(Processor):
    """Copies ''intro_indices'' and places it ''outro_indices''.

    # Arguments
        intro_indices: List of Ints.
        outro_indices: List of Ints.
    """
    def __init__(self, intro_indices, outro_indices):
        super(CopyDomain, self).__init__()
        if not isinstance(intro_indices, list):
            raise ValueError('``intro_indices`` must be a list')
        if not isinstance(outro_indices, list):
            raise ValueError('``outro_indices`` must be a list')
        self.intro_indices = intro_indices
        self.outro_indices = outro_indices

    def _select(self, inputs, indices):
        return [inputs[index] for index in indices]

    def _insert(self, args, axes, values):
        [args.insert(axis, value) for axis, value in zip(axes, values)]
        return args

    def call(self, *args):
        selections = self._select(args, self.intro_indices)
        args = self._insert(list(args), self.outro_indices, selections)
        return tuple(args)


class UnpackDictionary(Processor):
    """Unpacks dictionary into a tuple.
    # Arguments
        order: List of strings containing the keys of the dictionary.
            The order of the list is the order in which the tuple
            would be ordered.
    """
    def __init__(self, order):
        if not isinstance(order, list):
            raise ValueError('``order`` must be a list')
        self.order = order
        super(UnpackDictionary, self).__init__()

    def call(self, kwargs):
        args = tuple([kwargs[name] for name in self.order])
        return args


class WrapOutput(Processor):
    """Wraps arguments in dictionary

    # Arguments
        keys: List of strings representing the keys used to wrap the inputs.
            The order of the list must correspond to the same order of
            inputs (''args'').
    """
    def __init__(self, keys):
        if not isinstance(keys, list):
            raise ValueError('``order`` must be a list')
        self.keys = keys
        super(WrapOutput, self).__init__()

    def call(self, *args):
        return dict(zip(self.keys, args))


class ExtendInputs(Processor):
    """Extends number of inputs a function can take applying the identity
    function to all new/extended inputs.
    e.g. For a given function f(x) = y. If g = ExtendInputs(f), we can
    now have g(x, x1, x2, ..., xn) = y, x1, x2, ..., xn.

    # Arguments
        processor: Function e.g. any procesor in ''paz.processors''.
    """
    def __init__(self, processor):
        self.processor = processor
        name = '-'.join([self.__class__.__name__, self.processor.name])
        super(ExtendInputs, self).__init__(name)

    def call(self, X, *args):
        return self.processor(X), args


class Concatenate(Processor):
    """Concatenates a list of arrays in given ''axis''.

    # Arguments
        axis: Int.
    """
    def __init__(self, axis):
        super(Concatenate, self)
        self.axis = axis

    def call(self, inputs):
        return np.concatenate(inputs, self.axis)


class SequenceWrapper(Processor):
    """Wraps arguments to directly use
    ''paz.abstract.ProcessingSequence'' or
    ''paz.abstract.GeneratingSequence''.

    # Arguments
        inputs_info: Dictionary containing an integer per key representing
            the argument to grab, and as value a dictionary containing the
            tensor name as key and the tensor shape of a single sample as value
            e.g. {0: {'input_image': [300, 300, 3]}, 1: {'depth': [300, 300]}}.
            The values given here are for the inputs of the model.
        labels_info: Dictionary containing an integer per key representing
            the argument to grab, and as value a dictionary containing the
            tensor name as key and the tensor shape of a single sample as value
            e.g. {2: {'classes': [10]}}.
            The values given here are for the labels of the model.
    """
    def __init__(self, inputs_info, labels_info):
        if not isinstance(inputs_info, dict):
            raise ValueError('``inputs_info`` must be a dictionary')
        self.inputs_info = inputs_info
        if not isinstance(labels_info, dict):
            raise ValueError('``inputs_info`` must be a dictionary')
        self.labels_info = labels_info
        self.inputs_name_to_shape = self._extract_name_to_shape(inputs_info)
        self.labels_name_to_shape = self._extract_name_to_shape(labels_info)
        self.ordered_input_names = self._extract_ordered_names(inputs_info)
        self.ordered_label_names = self._extract_ordered_names(labels_info)
        super(SequenceWrapper, self).__init__()

    def _extract_name_to_shape(self, info):
        name_to_shape = {}
        for values in info.values():
            for key, value in values.items():
                name_to_shape[key] = value
        return name_to_shape

    def _extract_ordered_names(self, info):
        arguments = list(info.keys())
        arguments.sort()
        names = []
        for argument in arguments:
            names.append(list(info[argument].keys())[0])
        return names

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
    """Perform input preprocessing, model prediction and output postprocessing.

    # Arguments
        model: Class with a ''predict'' method e.g. a Keras model.
        preprocess: Function applied to given inputs.
        postprocess: Function applied to outputted predictions from model.
    """
    def __init__(self, model, preprocess=None, postprocess=None):
        super(Predict, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def call(self, x):
        return predict(x, self.model, self.preprocess, self.postprocess)


class ToClassName(Processor):
    def __init__(self, labels):
        super(ToClassName, self).__init__()
        self.labels = labels

    def call(self, x):
        return self.labels[np.argmax(x)]


class ExpandDims(Processor):
    """Expand dimension of given array.

    # Arguments
        axis: Int.
    """
    def __init__(self, axis):
        super(ExpandDims, self).__init__()
        self.axis = axis

    def call(self, x):
        return np.expand_dims(x, self.axis)


class SelectElement(Processor):
    """Selects element of input value.

    # Arguments
        index: Int. argument to select from ''inputs''.
    """
    def __init__(self, index):
        super(SelectElement, self).__init__()
        self.index = index

    def call(self, inputs):
        return inputs[self.index]


class BoxClassToOneHotVector(Processor):
    """Transform box data with class index to a one-hot encoded vector.

    # Arguments
        num_classes: Integer. Total number of classes.
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
    """Copies value passed to function.
    """
    def __init__(self):
        super(Copy, self).__init__()

    def call(self, x):
        return x.copy()


class Lambda(object):
    """Applies a lambda function as a processor transformation.

    # Arguments
        function: Function.
    """

    def __init__(self, function):
        self.function = function

    def __call__(self, x):
        return self.function(x)


class StochasticProcessor(Processor):
    def __init__(self, probability=0.5, name=None):
        """Adds stochasticity to the user implemented ``call`` function

        # Arguments:
            probability: Probability of calling ``call`` function

        # Example:
        ```python
        class RandomAdd(StochasticProcessor):
        def __init__(self, probability=0.5):
            super(StochasticProcessor, self).__init__(probability)

        def call(self, x):
            return x + 1

        random_add = RandomAdd(probability=0.5)
        # value can be either 1.0 or 2.0
        value = random_add(1.0)
        ```
        """
        super(StochasticProcessor, self).__init__(name=name)
        self.probability = probability

    def call(self, X):
        raise NotImplementedError

    def __call__(self, X):
        if self.probability >= np.random.rand():
            return self.call(X)
        return X


class Stochastic(Processor):
    def __init__(self, function, probability=0.5, name=None):
        """Adds stochasticity to a given ``function``

        # Arguments:
            function: Callable object i.e. python function or
                ``paz.abstract.Processor``.
            probability: Probability of calling ``function``.

        # Example:
        ```python
        stochastic_add_one = Stochastic(lambda x: x + 1, 0.5)
        # value can be either 0.0 or 1.0
        value = random_add(0.0)
        ```
        """
        super(Stochastic, self).__init__(name=name)
        self.function = function
        self.probability = probability

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, probability):
        assert 0.0 <= probability <= 1.0, 'Probability must be between 0 and 1'
        self._probability = probability

    def call(self, X):
        if self.probability >= np.random.rand():
            return self.function(X)
        return X


class UnwrapDictionary(Processor):
    """Unwraps a dictionry into a list given the key order.
    """
    def __init__(self, keys):
        super(UnwrapDictionary, self).__init__()
        self.keys = keys

    def call(self, dictionary):
        return [dictionary[key] for key in self.keys]


class Scale(Processor):
    """Scales an input.
    """
    def __init__(self, scales):
        super(Scale, self).__init__()
        self.scales = scales

    def call(self, values):
        return self.scales * values


class AppendValues(Processor):
    """Append dictionary values to lists

    # Arguments
        keys: Keys to dictionary values
    """
    def __init__(self, keys):
        super(AppendValues, self).__init__()
        self.keys = keys

    def call(self, dictionary, lists):
        return append_values(dictionary, lists, self.keys)


class BooleanToTextMessage(Processor):
    """Convert a boolean to text message.
    # Arguments
        true_message: String. Message for true case.
        false_message: String. Message for false case.
        Flag: Boolean.

    # Returns
        message: String.
    """
    def __init__(self, true_message, false_message):
        super(BooleanToTextMessage, self).__init__()
        self.true_message = true_message
        self.false_message = false_message

    def call(self, flag):
        if flag:
            message = self.true_message
        else:
            message = self.false_message
        return message


class PrintTopics(Processor):
    """Prints topics
    # Arguments
        topics: List of keys to the inputted dictionary

    # Returns
        Returns same dictionary but outputs to terminal topic values.
    """
    def __init__(self, topics):
        super(PrintTopics, self).__init__()
        self.topics = topics

    def call(self, dictionary):
        [print(dictionary[topic]) for topic in self.topics]
        return dictionary
