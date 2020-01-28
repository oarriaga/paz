import os
import h5py
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Progbar


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
