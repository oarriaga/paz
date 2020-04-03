from paz.core import Processor
from paz.core import SequentialProcessor
import numpy as np


class ProcessorA(Processor):
    def __init__(self):
        super(ProcessorA, self).__init__()

    def call(self, image, boxes):
        boxes = boxes - 1.0
        return image, boxes


class ProcessorB(Processor):
    def __init__(self):
        super(ProcessorB, self).__init__()

    def call(self, image, boxes):
        boxes = boxes - 2.0
        return image, boxes


class ProcessorC(Processor):
    def __init__(self, probability=0.5):
        super(ProcessorC, self).__init__()

    def call(self, image):
        return image / 255.0


class TransformA(SequentialProcessor):
    def __init__(self):
        super(TransformA, self).__init__()
        self.add(ProcessorC())


class TransformB(SequentialProcessor):
    def __init__(self):
        super(TransformB, self).__init__()
        self.add(ProcessorA())
        self.add(ProcessorB())
        self.add(ProcessorB())


class TransformC(SequentialProcessor):
    def __init__(self):
        super(TransformB, self).__init__()
        self.add(ProcessorA())


def test_arg_in_sequential_processor_input():
    transformA = TransformA()
    values = transformA(255.0)
    assert np.isclose(values == 1.0)


def test_kwargs_in_sequential_processor_input():
    transformB = TransformB()
    values = transformB(image=1.0, boxes=2.0)
    assert np.allclose([1.0, -3.0], values)


def test_kwargs_invariance_in_sequential_processor_input():
    transformB = TransformB()
    values = transformB(boxes=2.0, image=1.0)
    assert np.allclose([1.0, -3.0], values)


def test_flipped_kwargs_in_sequential_processor_input():
    transformB = TransformB()
    values = transformB(boxes=1.0, image=2.0)
    assert np.allclose([2.0, -4.0], values)
