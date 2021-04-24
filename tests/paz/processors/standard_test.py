from paz.abstract import SequentialProcessor, Processor
from paz.processors import ControlMap, StochasticProcessor, Stochastic


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


class RandomAdd(StochasticProcessor):
    def __init__(self, probability=0.5):
        super(RandomAdd, self).__init__(probability)

    def call(self, x):
        return x + 1


class NormalAdd(Processor):
    def __init__(self):
        super(NormalAdd, self).__init__()

    def call(self, x):
        return x + 1


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


def test_maximum_probability():
    random_add = RandomAdd(probability=1.0)
    assert random_add(1.0) == 2.0


def test_minimum_probability():
    random_add = RandomAdd(probability=0.0)
    assert random_add(1.0) == 1.0


def test_stochastic_in_sequential_processor():
    function = SequentialProcessor()
    function.add(RandomAdd(probability=1.0))
    function.add(RandomAdd(probability=1.0))
    assert function(2.0) == 4.0


def test_stochastic_and_deterministic_in_sequential_processor():
    function = SequentialProcessor()
    function.add(RandomAdd(probability=1.0))
    function.add(NormalAdd())
    assert function(2.0) == 4.0


def test_deterministic_and_stochastic_in_sequential_processor():
    function = SequentialProcessor()
    function.add(NormalAdd())
    function.add(RandomAdd(probability=1.0))
    assert function(2.0) == 4.0


def test_stochastic_processor_with_max_probability():
    class AddOne(Processor):
        def __init__(self):
            super(Processor, self).__init__()

        def call(self, x):
            return x + 1

    stochastic_add_one = Stochastic(AddOne(), 1.0)
    assert stochastic_add_one(10.0) == 11.0


def test_stochastic_functions_with_max_probability():
    stochastic_add_one = Stochastic(lambda x: x + 1.0, 1.0)
    assert stochastic_add_one(10.0) == 11.0


def test_stochastic_processor_with_min_probability():
    class AddOne(Processor):
        def __init__(self):
            super(Processor, self).__init__()

        def call(self, x):
            return x + 1

    stochastic_add_one = Stochastic(AddOne(), 0.0)
    assert stochastic_add_one(10.0) == 10.0


def test_stochastic_functions_with_min_probability():
    stochastic_add_one = Stochastic(lambda x: x + 1.0, 0.0)
    assert stochastic_add_one(10.0) == 10.0


def test_stochastic_functions_with_probability():
    stochastic_add_one = Stochastic(lambda x: x + 1.0, 0.5)
    for arg in range(100):
        assert stochastic_add_one(10.0) in [10.0, 11.0]


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
