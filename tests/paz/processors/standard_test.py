from paz.abstract import SequentialProcessor, Processor
# from paz.processors import ControlMap


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


class ControlMap(Processor):
    """Controls which inputs are passed ''processor'' and the order of its
        outputs.

    # Arguments
        processor: Function e.g. a ''paz.processor''
        intro_indices: List of Ints.
        outro_indices: List of Ints.
    """
    def __init__(self, processor, intro_indices=[0], outro_indices=[0], keep=None):
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
        if len(processed_args) != outro_indices:
            raise ValueError("Mismatch of ``outro_indices`` "
                             "and processor's output")
        if keep is not None:
            keep_args = self._select(selected_args, list(keep.keys())

        args = self._insert(remaining_args, processed_args, self.outro_indices)
        return tuple(args)


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


test_controlmap_reduction_and_selection_to_arg_1()
test_controlmap_reduction_and_selection_to_arg_2()
test_controlmap_reduction_and_flip()
test_controlmap_reduction_and_retention()
test_controlmap_parallelization()
test_controlmap_parallelization_in_different_order()

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
