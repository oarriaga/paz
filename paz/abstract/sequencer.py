from tensorflow.keras.utils import Sequence
import numpy as np


class ProcessingSequencer(Sequence):
    """Base sequencer class for processing or generating batches.
    If data is ``None`` the sequencer assumes that ``processor``
        generates the data. If data is not ``None`` the sequencer
        assumes the ``processor`` works as data processing pipeline.
    # Arguments
        processor: Function. If data is not ``None``, ``processor``
            takes a sample (see data) as input and returns a dictionary
            with keys ``inputs`` and ``labels`` and values dictionaries
            with keys being the ``layer names`` in which the values
            (numpy arrays) will be inputted.
        batch_size: Int.
        data: List of dictionaries. The length of the list corresponds to the
            amount of samples in the data. Inside each sample there should
            be a dictionary with `keys` indicating the data types/topics
            e.g. ``image``, ``depth``, ``boxes`` and as `values` of these
            `keys` the corresponding data e.g. strings, numpy arrays, etc.
    """
    def __init__(self, processor, batch_size, data):
        self.processor = processor
        self.input_topics = self.processor.processors[-1].input_topics
        self.label_topics = self.processor.processors[-1].label_topics
        self.batch_size = batch_size
        self.data = data

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, batch_index):
        batch_arg_A = self.batch_size * (batch_index)
        batch_arg_B = self.batch_size * (batch_index + 1)
        batch = self.data[batch_arg_A:batch_arg_B]
        inputs_batch = self.get_empty_batch(
            self.input_topics, self.processor.input_shapes)
        labels_batch = self.get_empty_batch(
            self.label_topics, self.processor.label_shapes)
        for sample_arg, unprocessed_sample in enumerate(batch):
            sample = self.processor(unprocessed_sample.copy())
            for topic, data in sample['inputs'].items():
                inputs_batch[topic][sample_arg] = data
            for topic, data in sample['labels'].items():
                labels_batch[topic][sample_arg] = data
        return inputs_batch, labels_batch

    def get_empty_batch(self, topics, shapes):
        batch = {}
        for topic, shape in zip(topics, shapes):
            batch[topic] = np.zeros((self.batch_size, *shape))
        return batch


class GeneratingSequencer(Sequence):
    """Base sequencer class for processing or generating batches.
    If data is ``None`` the sequencer assumes that ``processor``
        generates the data. If data is not ``None`` the sequencer
        assumes the ``processor`` works as data processing pipeline.
    # Arguments
        processor: Function. If data is not ``None``, ``processor``
            takes a sample (see data) as input and returns a dictionary
            with keys ``inputs`` and ``labels`` and values dictionaries
            with keys being the ``layer names`` in which the values
            (numpy arrays) will be inputted.
        batch_size: Int.
        data: List of dictionaries. The length of the list corresponds to the
            amount of samples in the data. Inside each sample there should
            be a dictionary with `keys` indicating the data types/topics
            e.g. ``image``, ``depth``, ``boxes`` and as `values` of these
            `keys` the corresponding data e.g. strings, numpy arrays, etc.
    """
    def __init__(self, processor, batch_size=32, as_list=False, num_steps=100):
        self.processor = processor
        self.input_topics = self.processor.processors[-1].input_topics
        self.label_topics = self.processor.processors[-1].label_topics
        self.batch_size = batch_size
        self.as_list = as_list
        self.num_steps = num_steps

    def __len__(self):
        return self.num_steps

    def __getitem__(self, batch_index):
        inputs_batch = self.get_empty_batch(
            self.input_topics, self.processor.input_shapes)
        labels_batch = self.get_empty_batch(
            self.label_topics, self.processor.label_shapes)
        for sample_arg in range(self.batch_size):
            sample = self.processor({'image': None})
            for topic, data in sample['inputs'].items():
                inputs_batch[topic][sample_arg] = data
            for topic, data in sample['labels'].items():
                labels_batch[topic][sample_arg] = data
        if self.as_list:
            inputs_batch = self.to_list(inputs_batch, self.input_topics)
            labels_batch = self.to_list(labels_batch, self.label_topics)
        return inputs_batch, labels_batch

    def get_empty_batch(self, topics, shapes):
        batch = {}
        for topic, shape in zip(topics, shapes):
            batch[topic] = np.zeros((self.batch_size, *shape))
        return batch

    def to_list(self, batch, topics):
        return [batch[topic] for topic in topics]
