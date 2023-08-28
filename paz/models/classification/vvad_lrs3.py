import tensorflow as tf

keras = tf.keras
from keras.models import Model
from keras.layers import (Input, BatchNormalization, Flatten, Dense, LSTM, TimeDistributed)
from keras.applications.mobilenet import MobileNet

def VVAD_LRS3_LSTM(weights=None, input_shape=(38, 96, 96, 3)):
    """Binary Classification for videos with 2+1D CNNs.
    # Arguments
        weights: String, path to the weights file to load. TODO add weights implementation when weights are available
        input_shape: List of integers. Input shape to the model in following format: (frames, height, width, channels)
        e.g. (38, 96, 96, 3).

    # Reference
        - [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248v3)
        - [Video classification with a 3D convolutional neural network]
        (https://www.tensorflow.org/tutorials/video/video_classification#load_and_preprocess_video_data)


        Model params according to vvadlrs3.pretrained_models.getFaceImageModel().summary()
    """
    if len(input_shape) != 4:
        raise ValueError(
            '`input_shape` must be a tuple of 4 integers. '
            'Received: %s' % (input_shape,))

    # initializer_glorot = tf.keras.initializers.GlorotUniform(seed=305865)
    # initializer_orthogonal = tf.keras.initializers.Orthogonal(seed=183319)

    # input_shape = (None, 10, HEIGHT, WIDTH, 3)
    image = Input(shape=input_shape, name='image')
    x = image

    base_model = MobileNet(
        weights="imagenet", include_top=False, input_shape=input_shape[1:])

    flatten = Flatten()(base_model.output)
    base_model = Model(base_model.input, flatten)
    x = TimeDistributed(base_model)(x)

    x = LSTM(32)(x)
    x = BatchNormalization()(x)

    # Add some more dense here
    for i in range(1):
        x = Dense(512, activation='relu')(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=image, outputs=x, name='Vvad_lrs3')

    return model
