import os
import h5py
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Progbar


class PerPixelActionScore(Callback):
    def __init__(self, sequencer, topic, evaluator, shape, epochs, filepath):
        super(Callback, self).__init__()
        self.sequencer = sequencer
        self.topic = topic
        self.evaluator = evaluator
        self.epochs = epochs
        H, W = self.shape = shape
        self.filepath = filepath
        self._build_directory(filepath)

        self.write_file = h5py.File(filepath, 'w')
        self.action_scores = self.write_file.create_dataset(
            'action_scores', (self.epochs, self.num_samples, H, W))

    def _build_directory(self, filepath):
        directory_name = os.path.dirname(filepath)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

    @property
    def num_samples(self):
        return len(self.sequencer) * self.sequencer.batch_size

    @property
    def batch_size(self):
        return self.sequencer.batch_size

    def on_epoch_end(self, epoch, logs=None):
        print('\n Computing per-pixel evaluations for epoch', epoch)
        progress_bar = Progbar(len(self.sequencer))
        for batch_index in range(len(self.sequencer)):
            inputs, labels = self.sequencer.__getitem__(batch_index)
            batch_arg_A = self.batch_size * (batch_index)
            batch_arg_B = self.batch_size * (batch_index + 1)
            y_true = labels[self.topic]
            y_pred = self.model(inputs)
            score = self.evaluator(y_true, y_pred)
            self.action_scores[epoch, batch_arg_A:batch_arg_B, :, :] = score
            progress_bar.update(batch_index + 1)
        self.action_scores.flush()

    def on_train_end(self, logs=None):
        print('\n Closing writing file in ', self.filepath)
        self.write_file.close()
