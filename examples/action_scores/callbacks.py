import os

import h5py
import numpy as np
import keras
from keras.callbacks import Callback


class ScalarActionScore(Callback):
    """Records per-sample evaluation scores each epoch to estimate sample
    difficulty (action scores). Writes an HDF5 dataset of shape
    `(epochs, num_samples, num_evaluators)`."""

    def __init__(self, sequencer, evaluators, epochs, filepath):
        super().__init__()
        self.sequencer = sequencer
        self.evaluators = evaluators
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.write_file = h5py.File(filepath, "w")
        shape = (epochs, self.num_samples, len(evaluators))
        self.evaluations = self.write_file.create_dataset("evaluations", shape)

    @property
    def batch_size(self):
        return self.sequencer.batch_size

    @property
    def num_samples(self):
        return len(self.sequencer) * self.sequencer.batch_size

    def on_epoch_end(self, epoch, logs=None):
        for batch_arg in range(len(self.sequencer)):
            inputs, labels = self.sequencer[batch_arg]
            predictions = self.model(inputs)
            start = self.batch_size * batch_arg
            stop = self.batch_size * (batch_arg + 1)
            for evaluator_arg, evaluate in enumerate(self.evaluators):
                scores = np.asarray(evaluate(labels, predictions))
                self.evaluations[epoch, start:stop, evaluator_arg] = scores
        self.evaluations.flush()

    def on_train_end(self, logs=None):
        self.write_file.close()


class FeatureExtractor(Callback):
    """Extracts features from `layer_name` for every sample on train end,
    writing an HDF5 dataset of shape `(num_samples, num_features)`. Spatial
    feature maps are global-average-pooled to a single vector per sample."""

    def __init__(self, layer_name, sequencer, filepath):
        super().__init__()
        self.layer_name = layer_name
        self.sequencer = sequencer
        self.filepath = filepath

    @property
    def batch_size(self):
        return self.sequencer.batch_size

    @property
    def num_samples(self):
        return len(self.sequencer) * self.sequencer.batch_size

    def on_train_end(self, logs=None):
        layer = self.model.get_layer(self.layer_name)
        extractor = keras.Model(self.model.input, layer.output)
        num_features = layer.output.shape[-1]
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with h5py.File(self.filepath, "w") as write_file:
            features = write_file.create_dataset(
                "features", (self.num_samples, num_features))
            for batch_arg in range(len(self.sequencer)):
                inputs = self.sequencer[batch_arg][0]
                start = self.batch_size * batch_arg
                stop = self.batch_size * (batch_arg + 1)
                features[start:stop] = pool_features(extractor(inputs))


def pool_features(features):
    features = np.asarray(features)
    if features.ndim == 4:
        features = features.mean(axis=(1, 2))
    return features
