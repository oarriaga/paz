from tensorflow.keras.utils import Sequence
import numpy as np


class Sequencer(Sequence):
    """Abstract base sequencer class for dispatching batches.

    # Arguments
        data: List of specific dataset samples.
        batch_size: Integer. Number of samples returned
            after calling __getitem__
    # Methods
        _preprocess_sample()
    """
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def _preprocess_sample(self, sample):
        """Process sample features and labels.

        # Arguments
            sample: List of dictionaries containing 'inputs' and
                'targets' as keys.

        # Returns
            Two elements containing preprocessed features and labels.
        """
        raise NotImplementedError

    def __getitem__(self, batch_index):
        batch_arg_0 = self.batch_size * (batch_index)
        batch_arg_1 = self.batch_size * (batch_index + 1)
        batch_samples = self.data[batch_arg_0:batch_arg_1]
        inputs, labels = [], []
        for sample in batch_samples:
            input_sample, label_sample = self._preprocess_sample(sample)
            inputs.append(input_sample)
            labels.append(label_sample)
        labels = np.asarray(labels)
        inputs = np.asarray(inputs)
        return inputs, labels
