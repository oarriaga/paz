from paz.abstract.sequence import SequenceExtra
import numpy as np


class ProcessingSequence(SequenceExtra):
    """Sequence generator used for generating samples.

    # Arguments
        processor: Function used for generating and processing ``samples``.
        batch_size: Int.
        num_steps: Int. Number of steps for each epoch.
        as_list: Bool, if True ``inputs`` and ``labels`` are dispatched as
            lists. If false ``inputs`` and ``labels`` are dispatched as
            dictionaries.
    """
    def __init__(self, processor, batch_size, data, num_steps, as_list=False):
        self.num_steps = num_steps
        super(ProcessingSequence, self).__init__(
            processor, batch_size, as_list)
        self.data = data

    def _num_batches(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __len__(self):
        return self.num_steps

    def process_batch(self, inputs, labels, batch_index):
        unprocessed_batch = self._get_unprocessed_batch(self.data, batch_index)

        for sample_arg, unprocessed_sample in enumerate(unprocessed_batch):
            sample = self.pipeline(unprocessed_sample.copy())
            self._place_sample(sample['inputs'], sample_arg, inputs)
            self._place_sample(sample['labels'], sample_arg, labels)
        return inputs, labels

    def _get_unprocessed_batch(self, data, batch_index):
        # batch_index = np.random.randint(0, self._num_batches())
        batch_index = 0
        batch_arg_A = self.batch_size * (batch_index)
        batch_arg_B = self.batch_size * (batch_index + 1)
        unprocessed_batch = data[batch_arg_A:batch_arg_B]
        return unprocessed_batch
