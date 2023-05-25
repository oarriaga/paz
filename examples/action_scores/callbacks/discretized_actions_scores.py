import os
import h5py
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Progbar


class DiscretizedActionScores(Callback):
    def __init__(self, sequencer, label_topic, discretized_topic, num_bins,
                 evaluators, epochs, filepath):

        self.sequencer = sequencer
        self.label_topic = label_topic
        self.evaluators = evaluators
        self.epochs = epochs
        self.filepath = filepath
        self.discretized_topic = discretized_topic
        self.num_bins = num_bins

        directory_name = os.path.dirname(self.filepath)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        self.write_file = h5py.File(self.filepath, 'w')
        self.evaluations = self.write_file.create_dataset(
            'evaluations',
            (3, self.num_bins, self.epochs, self.num_evaluators))

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
                y_true = self.model(inputs['image'])
                y_pred = labels[self.label_topic]
                args = inputs[self.discretized_topic].astype('int')
                evaluation = evaluator(y_true, y_pred)
                evaluation = np.mean(evaluation, axis=(1, 2))
                # fancy indexing not supported in h5py
                for enu, arg in enumerate(args[:, 0]):
                    self.evaluations[0, arg, epoch, eval_arg] = evaluation[enu]
                for enum, arg in enumerate(args[:, 1]):
                    self.evaluations[1, arg, epoch, eval_arg] = evaluation[enu]
                for enum, arg in enumerate(args[:, 2]):
                    self.evaluations[2, arg, epoch, eval_arg] = evaluation[enu]
            progress_bar.update(batch_index + 1)
        self.evaluations.flush()

    def on_train_end(self, logs=None):
        print('\n Closing writing file in ', self.filepath)
        self.write_file.close()
