from paz.abstract.processor import Processor, SequentialProcessor
import pytest

@pytest.fixture
def seq_processor():
    return SequentialProcessor()

@pytest.fixture(params = ['processor_0', 'processor_1'])
def processor(request):
    return Processor(request.param)

@pytest.fixture
def get_seq_processor():
    def _method(name):
        seq_ = SequentialProcessor()
        seq_.add(Processor(name))
        return seq_
    return _method

def test_add_processor(processor, seq_processor):
    if(isinstance(processor, type(Processor()))):
        seq_processor.add(processor)
        names = []
        for processor_ in seq_processor.processors:
            names.append(processor_.name)
        assert(processor.name in names)
    else:
        raise ValueError("Value is not an object of Processor()", type(processor))

@pytest.mark.parametrize("processor_name",['processor_0', 'processor_1'])
def test_remove_processor(get_seq_processor, processor_name):
    test_processor = get_seq_processor(processor_name)
    for proc_ in test_processor.processors:
        if(proc_.name == processor_name):
            test_processor.remove(processor_name)
            break
        else:
            print("Processor %s not in seq_processor", processor_name)
    for processor_ in test_processor.processors:
        assert(processor_.name != processor_name)

@pytest.mark.parametrize("index",[0, 1])
def test_insert_processor(index, get_seq_processor, processor):
    if(isinstance(index, int)):
        sq_processor = get_seq_processor('processor_1')
        if(index < (len(sq_processor.processors) - 1)):
            raise IndexError("Index out of bounds")
        else:
            sq_processor.insert(index, processor)
            inserted_processor = sq_processor.processors[index].name
        assert(processor.name == inserted_processor)
    else:
        raise ValueError("Index is not of type int")

@pytest.mark.parametrize("index, names",[(0, 'processor_0')])
def test_pop_processor(index, names, get_seq_processor):
    if(isinstance(index, int)):
        sq_processor = get_seq_processor(names)
        if(index < (len(sq_processor.processors) - 1)):
            raise IndexError("Index out of bounds")
        else:
            to_pop = sq_processor.processors[index].name
            poped_proc = sq_processor.pop(index)
            for proc in sq_processor.processors:
                assert(to_pop != proc.name)
    else:
        raise ValueError("Index is not of type int")