from paz.abstract import ProcessingSequence, Processor, SequentialProcessor
from paz import processors as pr
import numpy as np


class SumOne(Processor):
    def __init__(self):
        super(SumOne, self).__init__()

    def call(self, image, value_B):
        return value_A + 1.0, value_B + 1.0


class RandomFlipBoxesLeftRight(Processor):
    def __init__(self):
        super(RandomFlipBoxesLeftRight, self).__init__()

    def call(self, image, boxes):
        if np.random.randint(0, 2):
            width = image.shape[1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            image = image[:, ::-1]
        return image, boxes


data = [{'value_A': np.ones((10, 10)),
         'value_B': np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]])}]
processor = SequentialProcessor()
processor.add(pr.UnpackDictionary(['value_A', 'value_B']))
# processor.add(SumOne())
processor.add(RandomFlipBoxesLeftRight())
processor.add(pr.SequenceWrapper(
    {0: {'value_A': [10, 10]}},
    {1: {'value_B': [2, 3]}}))
sequence = ProcessingSequence(processor, 1, data)

for _ in range(10):
    batch = sequence.__getitem__(0)
    value_A, value_B = batch[0]['value_A'][0], batch[1]['value_B'][0]
    print(value_A, value_B)
