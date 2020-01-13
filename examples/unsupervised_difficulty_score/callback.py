import os
import numpy as np
import h5py
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Progbar


class DifficultyCallback(Callback):
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
