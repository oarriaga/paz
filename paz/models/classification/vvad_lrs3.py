import tensorflow as tf

keras = tf.keras
from keras.models import Model
from keras.layers import (Input, BatchNormalization, Flatten, Dense, LSTM, TimeDistributed)
from keras.applications.mobilenet import MobileNet
import random

def VVAD_LRS3_LSTM(weights=None, input_shape=(38, 96, 96, 3), seed=305865, tmp_weights_path="../../../../CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/vvad_lrs3-weights-57.hdf5"):
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

    # input_shape = (None, 10, HEIGHT, WIDTH, 3)
    image = Input(shape=input_shape, name='image')
    x = image

    base_model = MobileNet(
        weights=None, include_top=False, input_shape=input_shape[1:])

    flatten = Flatten()(base_model.output)
    base_model = Model(base_model.input, flatten)
    x = TimeDistributed(base_model)(x)

    x = LSTM(32, kernel_initializer=initializer_glorot_lstm, recurrent_initializer=initializer_orthogonal)(x)
    x = BatchNormalization()(x)

    # Add some more dense here
    for i in range(1):
        x = Dense(512, activation='relu', kernel_initializer=initializer_glorot_dense)(x)

    x = Dense(1, activation="sigmoid", kernel_initializer=initializer_glorot_output)(x)

    model = Model(inputs=image, outputs=x, name='VVAD_LRS3_LSTM')

    if weights == 'VVAD-LRS3':
        print("loading weights")
        model.load_weights(tmp_weights_path)      # TODO Replace with download link
    #     filename = 'fer2013_mini_XCEPTION.119-0.65.hdf5'
    #     path = get_file(filename, URL + filename, cache_subdir='paz/models')
    #     model = load_model(path)

    return model
