from tensorflow.keras.utils import Sequence
import numpy as np
from .processor import SequentialProcessor


class SequenceExtra(Sequence):
    def __init__(self, pipeline, batch_size, as_list=False):
        if not isinstance(pipeline, SequentialProcessor):
            raise ValueError('``processor`` must be a ``SequentialProcessor``')
        output_wrapper = pipeline.processors[-1]
        self.pipeline = pipeline
        self.inputs_name_to_shape = output_wrapper.inputs_name_to_shape
        self.labels_name_to_shape = output_wrapper.labels_name_to_shape
        self.batch_size = batch_size
        self.as_list = as_list

    def make_empty_batches(self, name_to_shape):
        batch = {}
        for name, shape in name_to_shape.items():
            batch[name] = np.zeros((self.batch_size, *shape))
        return batch

    def _to_list(self, batch, names):
        return [batch[name] for name in names]

    def _place_sample(self, sample, sample_arg, batch):
        for name, data in sample.items():
            batch[name][sample_arg] = data

    def _get_unprocessed_batch(self, data, batch_index):
        batch_arg_A = self.batch_size * (batch_index)
        batch_arg_B = self.batch_size * (batch_index + 1)
        unprocessed_batch = data[batch_arg_A:batch_arg_B]
        return unprocessed_batch

    def __getitem__(self, batch_index):
        inputs = self.make_empty_batches(self.inputs_name_to_shape)
        labels = self.make_empty_batches(self.labels_name_to_shape)
        inputs, labels = self.process_batch(inputs, labels, batch_index)
        if self.as_list:
            inputs = self._to_list(inputs, self.input_names)
            labels = self._to_list(labels, self.label_names)
        return inputs, labels

    def process_batch(self, inputs, labels, batch_index=None):
        raise NotImplementedError


class ProcessingSequence(SequenceExtra):
    def __init__(self, processor, batch_size, data, as_list=False):
        self.data = data
        super(ProcessingSequence, self).__init__(
            processor, batch_size, as_list)

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def process_batch(self, inputs, labels, batch_index):
        unprocessed_batch = self._get_unprocessed_batch(self.data, batch_index)
        
        for sample_arg, unprocessed_sample in enumerate(unprocessed_batch):
            sample = self.pipeline(unprocessed_sample.copy())
            self._place_sample(sample['inputs'], sample_arg, inputs)
            self._place_sample(sample['labels'], sample_arg, labels)
        return inputs, labels


class GeneratingSequence(SequenceExtra):
    def __init__(self, processor, batch_size, num_steps, as_list=False):
        self.num_steps = num_steps
        super(GeneratingSequence, self).__init__(
            processor, batch_size, as_list)

    def __len__(self):
        return self.num_steps

    def process_batch(self, inputs, labels, batch_index):
        for sample_arg in range(self.batch_size):
            sample = self.pipeline()
            self._place_sample(sample['inputs'], sample_arg, inputs)
            self._place_sample(sample['labels'], sample_arg, labels)
        return inputs, labels
