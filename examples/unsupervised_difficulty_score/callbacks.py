import os
import numpy as np
import h5py
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Progbar


class DifficultyTrackerBeta(Callback):
    def __init__(self, dataset, metrics_names, save_path, verbose=1):
        self.dataset = dataset
        self.save_path = save_path
        self.metrics_names = metrics_names
        self.verbose = verbose
        self.num_metrics = len(self.metrics_names)
        self.num_samples = len(dataset[0])
        self.save_path = os.path.join(save_path, 'metrics.hdf5')
        self.write_file = h5py.File(self.save_path, 'w')
        self.metrics = self.write_file.create_dataset(
            'metrics', (1, self.num_samples, self.num_metrics),
            maxshape=(None, self.num_samples, self.num_metrics),
            compression='gzip')

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0:
            new_shape = (self.metrics.shape[0] + 1, *self.metrics.shape[1:])
            self.metrics.resize(new_shape)

        inputs, labels = self.dataset
        if self.verbose:
            print('\n Computing per-sample metrics for epoch', epoch)
            progress_bar = Progbar(len(inputs))

        for sample_arg, (sample, target) in enumerate(zip(inputs, labels)):
            sample = np.expand_dims(sample, 0)
            target = np.expand_dims(target, 0)
            self.metrics[epoch, sample_arg, :] = self.model.evaluate(
                sample, target, verbose=0)
            if self.verbose:
                progress_bar.update(sample_arg + 1)
        self.metrics.flush()

    def on_train_end(self, logs=None):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        print('\n Saving metric in ', self.save_path)
        self.write_file.close()


class Evaluator(Callback):
    def __init__(self, sequencer, topic, evaluators, epochs, filename):
        self.sequencer = sequencer
        self.topic = topic
        self.evaluators = evaluators
        self.epochs = epochs
        self.filename = filename
        self.write_file = h5py.File(self.filename, 'w')
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
                y_true = self.model(inputs)
                y_pred = labels[self.topic]
                evaluation = evaluator(y_true, y_pred)
                self.evaluations[
                    epoch, batch_arg_A:batch_arg_B, eval_arg] = evaluation
            progress_bar.update(batch_index + 1)
        print('\n Finish per-sample evaluaiton for epoch', epoch)
        self.evaluations.flush()

    def on_train_end(self, logs=None):
        if not os.path.exists(self.filename):
            os.makedirs(self.filename)
        print('\n Saving metric in ', self.filename)
        self.write_file.close()
