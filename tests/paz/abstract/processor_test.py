from paz.abstract.processor import Processor, SequentialProcessor

def test_add_processor(sq_processor, processor):
    sq_processor.add(processor)
    names = []
    for processor_ in sq_processor.processors:
        names.append(processor_.name)
    assert(processor.name in names)

def test_remove_processor(seq_processor, processor):
    seq_processor.remove(processor.name)
    for processor_ in seq_processor.processors:
        assert(processor_.name != processor.name)

def test_insert_processor(index, seq_processor, processor):
    seq_processor.insert(index, processor)
    inserted_processor = seq_processor.processors[index].name
    assert(processor.name == inserted_processor)

def test_pop_processor(index, seq_processor):
    to_pop = seq_processor.processors[index].name
    poped_proc = seq_processor.pop(index)
    names = []
    for proc in seq_processor.processors:
        assert(to_pop != proc.name)

processor_0 = Processor('processor_0')
processor_1 = Processor('processor_1')
sq_processor = SequentialProcessor()

test_add_processor(sq_processor, processor_0)
test_add_processor(sq_processor, processor_1)

test_remove_processor(sq_processor, processor_1)
test_insert_processor(0, sq_processor, processor_1)
test_pop_processor(1, sq_processor)