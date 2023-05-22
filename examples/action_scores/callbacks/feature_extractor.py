import os
import h5py
import numpy as np

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.backend import function
from tensorflow.keras.utils import Progbar


class FeatureExtractor(Callback):
    def __init__(self, layer_name, sequencer, filepath):
        self.layer_name = layer_name
        self.sequencer = sequencer
        self.filepath = filepath

    @property
    def batch_size(self):
        return self.sequencer.batch_size

    @property
    def num_samples(self):
        return len(self.sequencer) * self.sequencer.batch_size

    def on_train_end(self, logs):
        print('Extracting features from layer:', self.layer_name)
        output_tensor = self.model.get_layer(self.layer_name).output
        feature_extractor = function(self.model.input, output_tensor)
        num_features = output_tensor.shape.as_list()[-1]

        directory_name = os.path.dirname(self.filepath)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        self.write_file = h5py.File(self.filepath, 'w')
        self.features = self.write_file.create_dataset(
            'features', (self.num_samples, num_features))

        progress_bar = Progbar(len(self.sequencer))
        for batch_index in range(len(self.sequencer)):
            inputs = self.sequencer.__getitem__(batch_index)[0]
            batch_arg_A = self.batch_size * (batch_index)
            batch_arg_B = self.batch_size * (batch_index + 1)
            features = feature_extractor(inputs)
            features = np.squeeze(features)
            self.features[batch_arg_A:batch_arg_B, :] = features
            progress_bar.update(batch_index + 1)
        self.write_file.close()
