import os
import h5py
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Progbar


class ScalarActionScore(Callback):
    """Estimates sample difficulty using action scores as described in [1].

    # Arguments
        sequencer: Generator. Keras ``Sequence`` for generating data samples.
        topic: String. Name of labels in sequencer.
        evaluators: List of callables for computing action scores.
        epochs: Int. Max number of epochs.
        filepath: String. Name of file to write action scores on.

    # References
        [1] [Action Scores](https://arxiv.org/pdf/2011.11461.pdf)
    """
    def __init__(self, sequencer, topic, evaluators, epochs, filepath):
        self.sequencer = sequencer
        self.topic = topic
        self.evaluators = evaluators
        self.epochs = epochs
        self.filepath = filepath

        directory_name = os.path.dirname(self.filepath)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        self.write_file = h5py.File(self.filepath, 'w')
        self.evaluations = self.write_file.create_dataset(
            'evaluations',
            (self.epochs, self.num_samples, self.num_evaluators))

    @property
    def num_evaluators(self):
        return len(self.evaluators)

    @property
    def num_samples(self):
        return len(self.sequencer) * self.sequencer.batch_size

    @property
    def batch_size(self):
        return self.sequencer.batch_size

    def on_epoch_end(self, epoch, logs=None):
        print('\n Computing per-sample evaluations for epoch', epoch)
        progress_bar = Progbar(len(self.sequencer))
        for batch_index in range(len(self.sequencer)):
            inputs, labels = self.sequencer.__getitem__(batch_index)
            for eval_arg, evaluator in enumerate(self.evaluators):
                batch_arg_A = self.batch_size * (batch_index)
                batch_arg_B = self.batch_size * (batch_index + 1)
                y_true = labels[self.topic]
                y_pred = self.model(inputs)
                evaluation = evaluator(y_true, y_pred)
                self.evaluations[
                    epoch, batch_arg_A:batch_arg_B, eval_arg] = evaluation
            progress_bar.update(batch_index + 1)
        self.evaluations.flush()

    def on_train_end(self, logs=None):
        print('\n Closing writing file in ', self.filepath)
        self.write_file.close()


class DataScalarActionScore(Callback):
    def __init__(self, data, evaluator, epochs, filepath):
        self.data = data
        self.evaluator = evaluator
        self.epochs = epochs
        self.filepath = filepath

        directory_name = os.path.dirname(self.filepath)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        self.write_file = h5py.File(self.filepath, 'w')
        self.evaluations = self.write_file.create_dataset(
            'evaluations', (self.epochs, self.num_samples, 1))

    @property
    def num_samples(self):
        return len(self.data[0])

    def on_epoch_end(self, epoch, logs=None):
        print('\n Computing per-sample evaluations for epoch', epoch)
        progress_bar = Progbar(self.num_samples)
        x_data, y_data = self.data
        for sample_arg, (x, y_true) in enumerate(zip(x_data, y_data)):
            y_pred = self.model(np.expand_dims(x, 0))
            y_true = np.expand_dims(y_true, [0, 1])
            evaluation = self.evaluator(y_true, y_pred)
            self.evaluations[epoch, sample_arg] = evaluation
            progress_bar.update(sample_arg + 1)
        self.evaluations.flush()

    def on_train_end(self, logs=None):
        print('\n Closing writing file in ', self.filepath)
        self.write_file.close()
