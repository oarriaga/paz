from paz.abstract.processor import Processor, SequentialProcessor
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
