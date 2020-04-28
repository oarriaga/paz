class Processor(object):
    """ Abstract class for creating a processor unit.

    # Arguments
        name: String indicating name of the processing unit.

    # Methods
        call()
    """
    def __init__(self, name=None):
        self.name = name
        self._process = self.call

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is None:
            name = self.__class__.__name__
        self._name = name

    def call(self, X):
        """Custom user's logic should be implemented here.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self._process(*args, **kwargs)


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

    def __call__(self, *args, **kwargs):
        # first call can take list or dictionary values.
        args = self.processors[0](*args, **kwargs)
        # further calls can be a tuple or single values.
        for processor in self.processors[1:]:
            if isinstance(args, tuple):
                args = processor(*args)
            else:
                args = processor(args)
        return args

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

    def insert(self, processor, index):
        """Inserts ``processor`` to self.processors queue at ``index``
        """
        return self.processors.insert(index)

    def get_processor(self, name):
        """Gets processor from sequencer
        # Arguments
            name: String indicating the process name
        """
        for processor in self.processors:
            if processor.name == name:
                return processor
