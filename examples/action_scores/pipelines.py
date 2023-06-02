import numpy as np
from paz import processors as pr


class OneHotVector(pr.Processor):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(OneHotVector, self).__init__()

    def call(self, label):
        one_hot_vector = np.zeros(self.num_classes)
        one_hot_vector[label] = 1.0
        return one_hot_vector


class ProcessImage(pr.SequentialProcessor):
    def __init__(self, size, num_classes, grayscale=True, hot_vector=False):
        super(ProcessImage, self).__init__()
        preprocess = pr.SequentialProcessor()
        preprocess.add(pr.ResizeImage((size, size)))
        preprocess.add(pr.CastImage(float))
        if grayscale:
            preprocess.add(pr.ExpandDims(axis=-1))
        preprocess.add(pr.NormalizeImage())

        self.add(pr.UnpackDictionary(['image', 'label']))
        self.add(pr.ControlMap(preprocess))
        if hot_vector:
            self.add(pr.ControlMap(OneHotVector(num_classes), [1], [1]))
        num_channels = 1 if grayscale else 3
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, num_channels]}},
            {1: {'label': [num_classes]}}))
