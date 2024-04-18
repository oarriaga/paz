import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, BatchNormalization, Flatten, Dense, LSTM, TimeDistributed)
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.utils import get_file
import random

URL = 'https://github.com/oarriaga/altamira-data/releases/download/v0.19/'

def VVAD_LRS3_LSTM(weights=None, input_shape=(38, 96, 96, 3), seed=305865):
    """Binary Classification for videos using a CNN based mobile net with an TimeDistributed layer (LSTM).
    # Arguments
        weights: ``None`` or string with pre-trained dataset. Valid datasets
            include only ``VVAD-LRS3``.
        input_shape: List of integers. Input shape to the model in following format: (frames, height, width, channels)
        e.g. (38, 96, 96, 3).
        seed: Integer. Seed for random number generator.

    # Reference
        - [The VVAD-LRS3 Dataset for Visual Voice Activity Detection]
        (https://api.semanticscholar.org/CorpusID:238198700)
        - [VVAD-LRS3 GitHub Repository]
        (https://github.com/adriandavidauer/VVAD)
    """
    if len(input_shape) != 4:
        raise ValueError(
            '`input_shape` must be a tuple of 4 integers. '
            'Received: %s' % (input_shape,))

    random.seed(seed)
    initializer_glorot_lstm = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    initializer_glorot_dense = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    initializer_glorot_output = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    initializer_orthogonal = tf.keras.initializers.Orthogonal(seed=random.randint(0, 1000000))

    image = Input(shape=input_shape, name='image')
    x = image

    base_model = MobileNet(
        weights=None, include_top=False, input_shape=input_shape[1:])

    flatten = Flatten()(base_model.output)
    base_model = Model(base_model.input, flatten)
    x = TimeDistributed(base_model)(x)

    x = LSTM(32, kernel_initializer=initializer_glorot_lstm, recurrent_initializer=initializer_orthogonal)(x)
    x = BatchNormalization()(x)

    x = Dense(512, activation='relu', kernel_initializer=initializer_glorot_dense)(x)

    x = Dense(1, activation="sigmoid", kernel_initializer=initializer_glorot_output)(x)

    model = Model(inputs=image, outputs=x, name='VVAD_LRS3_LSTM')

    if weights == 'VVAD_LRS3':
        print("loading weights")
        filename = 'vvad_lrs3-weights-23.hdf5'
        weights_path = get_file(filename, URL + filename, cache_subdir='paz/models')
        model.load_weights(weights_path)

    return model
