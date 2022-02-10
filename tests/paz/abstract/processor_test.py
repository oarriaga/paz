from paz.abstract.processor import Processor, SequentialProcessor
from paz import processors as pr
import numpy as np
import pytest


@pytest.fixture(params=['processor_0', 'processor_1'])
def processor(request):
    return Processor(request.param)


@pytest.fixture
def get_sequential_processor():
    def call(name):
        sequential_processor = SequentialProcessor()
        sequential_processor.add(Processor(name))
        return sequential_processor
    return call


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
    assert np.allclose(values, 1.0)


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


def test_add_processor(processor):
    sequential_processor = SequentialProcessor()
    if(isinstance(processor, type(Processor()))):
        sequential_processor.add(processor)
        assert(processor in sequential_processor.processors)
    else:
        raise ValueError('Value is not an object of Processor()')


@pytest.mark.parametrize('processor_name', ['processor_0', 'processor_1'])
def test_remove_processor(get_sequential_processor, processor_name):
    test_processor = get_sequential_processor(processor_name)
    for processor in test_processor.processors:
        if(processor.name == processor_name):
            test_processor.remove(processor_name)
            break
        else:
            print('Processor %s not in sequential_processor', processor_name)
    for processor in test_processor.processors:
        assert(processor.name != processor_name)


@pytest.mark.parametrize('index', [0, 1])
def test_insert_processor(index, get_sequential_processor, processor):
    if(isinstance(index, int)):
        sequential_processor = get_sequential_processor('processor_1')
        if(index < (len(sequential_processor.processors) - 1)):
            raise IndexError('Index out of bounds')
        else:
            sequential_processor.insert(index, processor)
            inserted_processor = sequential_processor.processors[index].name
        assert(processor.name == inserted_processor)
    else:
        raise ValueError('Index is not of type int')


@pytest.mark.parametrize('index, names', [(0, 'processor_0')])
def test_pop_processor(index, names, get_sequential_processor):
    if(isinstance(index, int)):
        sequential_processor = get_sequential_processor(names)
        if(index < (len(sequential_processor.processors) - 1)):
            raise IndexError('Index out of bounds')
        else:
            to_pop = sequential_processor.processors[index].name
            sequential_processor.pop(index)
            for processor in sequential_processor.processors:
                assert(to_pop != processor.name)
    else:
        raise ValueError('Index is not of type int')


class PipelineWithTwoChannels(SequentialProcessor):
    def __init__(self):
        super(PipelineWithTwoChannels, self).__init__()
        self.add(lambda x: x)
        self.add(pr.ControlMap(pr.Copy(), [0], [1], keep={0: 0}))


class PipelineWithThreeChannels(SequentialProcessor):
    def __init__(self):
        super(PipelineWithThreeChannels, self).__init__()
        self.add(lambda a, b: (a, b))
        self.add(pr.ControlMap(pr.Copy(), [0], [2], keep={0: 0}))


class PipelineWithThreeChannelsPlus(SequentialProcessor):
    def __init__(self):
        super(PipelineWithThreeChannelsPlus, self).__init__()
        self.add(lambda a, b: (a, b))
        self.add(pr.ControlMap(pr.Copy(), [0], [2], keep={0: 0}))
        self.add(pr.ControlMap(SumTwoValues(), [0, 1], [0]))


class SumTwoValues(Processor):
    def __init__(self):
        super(SumTwoValues, self).__init__()

    def call(self, A, B):
        return A + B


def test_copy_with_controlmap_using_2_channels():
    pipeline = PipelineWithTwoChannels()
    random_values = np.random.random((128, 128))
    values = pipeline(random_values)
    assert len(values) == 2
    assert np.allclose(values[0], random_values)
    assert np.allclose(values[1], random_values)


def test_copy_with_controlmap_using_3_channels():
    pipeline = PipelineWithThreeChannels()
    A_random_values = np.random.random((128, 128))
    B_random_values = np.random.random((128, 128))
    values = pipeline(A_random_values, B_random_values)
    assert len(values) == 3
    assert np.allclose(values[0], A_random_values)
    assert np.allclose(values[1], B_random_values)
    assert np.allclose(values[2], A_random_values)


def test_copy_with_controlmap_using_3_channels_plus():
    pipeline = PipelineWithThreeChannelsPlus()
    A_random_values = np.random.random((128, 128))
    B_random_values = np.random.random((128, 128))
    values = pipeline(A_random_values, B_random_values)
    assert len(values) == 2
    assert np.allclose(values[0], A_random_values + B_random_values)
    assert np.allclose(values[1], A_random_values)
