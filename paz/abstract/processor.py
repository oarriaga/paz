class Processor(object):
    """Abstract class for creating a processor unit.

    # Arguments
        name: String indicating name of the processing unit.

    # Methods
        call()

    # Example
    ```python
    class NormalizeImage(Processor):
    def __init__(self):
        super(NormalizeImage, self).__init__()

    def call(self, image):
        return image / 255.0
    ```

    # Why this name?
        Originally PAZ was only meant for pre-processing pipelines that
        included data-augmentation, normalization, etc. However, I found
        out that we could use the same API for post-processing; therefore,
        I thought at the time that ``Processor`` would be adequate to describe
        the capacity of both pre-processing and post-processing.
        Names that I also thought could have worked were: ``Function``,
        ``Functor`` but I didn't want to use those since I thought they could
        also cause confusion. Similarly, in Keras this abstraction is
        interpreted as a ``Layer`` but here I don't think that abstraction
        is adequate. A layer of computation maybe? So after having this
        thoughts swirling around I decided to go with ``Processor``
        and try to be explicit about my mental jugglery hoping the name
        doesn't cause much mental overhead.
    """
    def __init__(self, name=None):
        self.name = name

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
        return self.call(*args, **kwargs)


class SequentialProcessor(object):
    """Abstract class for creating a sequential pipeline of processors.

    # Arguments
        processors: List of instantiated child classes of ``Processor``
            classes.
        name: String indicating name of the processing unit.

    # Methods
        add()
        remove()
        pop()
        insert()
        get_processor()

    # Example
    ```python
    AugmentImage = SequentialProcessor()
    AugmentImage.add(pr.RandomContrast())
    AugmentImage.add(pr.RandomBrightness())
    augment_image = AugmentImage()

    transformed_image = augment_image(image)
    ```
    """
    def __init__(self, processors=None, name=None):
        self.processors = []
        if processors is not None:
            [self.add(processor) for processor in processors]
        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is None:
            name = self.__class__.__name__
        self._name = name

    def add(self, processor):
        """Adds a process to the sequence of processes to be applied to input.

        # Arguments
            processor: An instantiated child class of of ``Processor``.
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

    def insert(self, index, processor):
        """Inserts ``processor`` to self.processors queue at ``index``

        # Argument
            index: Int.
            processor: An instantiated child class of of ``Processor``.
        """
        return self.processors.insert(index, processor)

    def get_processor(self, name):
        """Gets processor from sequencer

        # Arguments
            name: String indicating the process name
        """
        for processor in self.processors:
            if processor.name == name:
                return processor
