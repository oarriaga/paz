from paz.abstract import SequentialProcessor, Processor
from paz.processors import ControlMap


class Sum(Processor):
    def __init__(self):
        super(Sum, self).__init__()

    def call(self, x, y):
        return x + y


class AddConstantToVector(Processor):
    def __init__(self, constant=5.0):
        self.constant = constant
        super(AddConstantToVector, self).__init__()

    def call(self, x, y):
        return x + self.constant, y + self.constant


class MultiplyByFactor(Processor):
    def __init__(self, factor=0.5):
        self.factor = factor
        super(MultiplyByFactor, self).__init__()

    def call(self, x):
        return self.factor * x


def test_controlmap_reduction_and_selection_to_arg_1():
    pipeline = SequentialProcessor()
    pipeline.add(ControlMap(Sum(), [1, 2], [0]))
    assert pipeline(2, 5, 10) == (5 + 10, 2)


def test_controlmap_reduction_and_selection_to_arg_2():
    pipeline = SequentialProcessor()
    pipeline.add(ControlMap(Sum(), [1, 2], [1]))
    assert pipeline(2, 5, 10) == (2, 5 + 10)


def test_controlmap_reduction_and_flip():
    pipeline = SequentialProcessor()
    pipeline.add(ControlMap(Sum(), [1, 2], [1]))
    pipeline.add(ControlMap(MultiplyByFactor(0.5), [0], [0]))
    pipeline.add(ControlMap(AddConstantToVector(0.1), [0, 1], [1, 0]))
    assert pipeline(2, 5, 10) == ((5 + 10) + 0.1, (2 * 0.5) + 0.1)


def test_controlmap_reduction_and_retention():
    pipeline = SequentialProcessor()
    pipeline.add(ControlMap(Sum(), [1, 5], [3]))
    assert pipeline(2, 5, 10, 6, 7, 8) == (2, 10, 6, 5 + 8, 7)


def test_controlmap_parallelization():
    pipeline = SequentialProcessor()
    pipeline.add(ControlMap(MultiplyByFactor(3.0), [0], [0]))
    pipeline.add(ControlMap(MultiplyByFactor(2.0), [1], [1]))
    assert pipeline(10, 5) == (10 * 3.0, 5 * 2.0)


def test_controlmap_parallelization_in_different_order():
    pipeline = SequentialProcessor()
    pipeline.add(ControlMap(MultiplyByFactor(2.0), [1], [1]))
    pipeline.add(ControlMap(MultiplyByFactor(3.0), [0], [0]))
    assert pipeline(10, 5) == (10 * 3.0, 5 * 2.0)


def test_controlmap_reduction_and_keep():
    pipeline = SequentialProcessor()
    pipeline.add(ControlMap(Sum(), [1, 2], [1], {2: 0}))
    assert pipeline(2, 5, 10) == (10, 2, 5 + 10)


# test_controlmap_reduction_and_selection_to_arg_1()
# test_controlmap_reduction_and_selection_to_arg_2()
# test_controlmap_reduction_and_flip()
# test_controlmap_reduction_and_retention()
# test_controlmap_parallelization()
# test_controlmap_parallelization_in_different_order()
# test_controlmap_reduction_and_keep()

'''
# sample = {'boxes': 5, 'keypoints': 10, 'image': 2}
# print(pipeline(**sample))
sample = (5, 10, 2)
print(pipeline(*sample))

# pipeline = SequentialProcessor()
# pipeline.add(ExtendInputs(MultiplyByFactor()))
# pipeline.add(Sum())
# print(pipeline(5, 5))
# print(pipeline(5, 5, 6))
'''
