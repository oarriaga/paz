import numpy as np


class Processor(object):
    """ Abstract class for creating a new processor unit.
    A processor unit logic lives in method `call` which is wrapped
    to work in two different modes: stochastic and deterministic.
    The stochastic mode is activated whenever a value different than
    `None` is given to the variable `probability`.
    If `None` is passed to `probability` the processor unit works
    deterministically.
    If the processor unit is working stochastically the logic in the
    method `call` will be applied to the input with the
    probability value given.
    It the processor is working deterministically the logic of the
    method `call` will be always applied.

    # Arguments
        probability: None or float between [0, 1]. See above for description.
        name: String indicating name of the processing unit.

    # Methods
        call()
    """
    def __init__(self, probability=None, name=None):
        self.probability = probability
        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is None:
            name = self.__class__.__name__
        self._name = name

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, probability):
        if probability is None:
            self._probability = None
            self._process = self.call
        elif (0.0 <= probability <= 1.0):
            self._probability = probability
            self._process = self.stochastic_process
        else:
            raise ValueError('Probability has to be between [0, 1]')

    def stochastic_process(self, kwargs):
        if np.random.random() < self.probability:
            kwargs = self.call(kwargs)
        return kwargs

    def call(self, kwargs):
        """ Logic to be implemented to transform kwargs
        """
        raise NotImplementedError

    def __call__(self, kwargs):
        return self._process(kwargs)


class SequentialProcessor(object):
    """ Abstract class for creating a sequential pipeline of processors.
    # Methods:
        add()
    """
    def __init__(self, processors=None):
        self.processors = []
        if processors is not None:
            [self.add(processor) for processor in processors]

    def add(self, processor):
        """ Adds a process to the sequence of processes to be applied to input.
        # Arguments
            processor: An extended class of the parent class `Process`.
        """
        self.processors.append(processor)

    def __call__(self, kwargs):
        for processor in self.processors:
            kwargs = processor(kwargs)
        return kwargs

    def remove(self, name):
        """Removes processor from sequence
        # Arguments
            name: String indicating the process name
        """
        for processor in self.processors:
            if processor.name == name:
                self.processors.remove(processor)

    def pop(self, index=-1):
        """Pops processor in given index from sequence
        # Arguments
            index: Int.
        """
        return self.processors.pop(index)

    def get_processor(self, name):
        """Gets processor from sequencer
        # Arguments
            name: String indicating the process name
        """
        for processor in self.processors:
            if processor.name == name:
                return processor
