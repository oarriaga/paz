from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, MaxPool2D, Flatten
from tensorflow.keras.layers import Conv2D, Dense

from layers import BilinearInterpolation
import numpy as np


def get_initial_weights(output_size):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights


def STN(input_shape=(60, 60, 1), interpolation_size=(30, 30), num_classes=10):
    image = Input(shape=input_shape)
    x = MaxPool2D(pool_size=(2, 2))(image)
    x = Conv2D(20, (5, 5))(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(20, (5, 5))(x)
    x = Flatten()(x)
    x = Dense(50)(x)
    x = Activation('relu')(x)
    x = Dense(6, weights=get_initial_weights(50))(x)
    interpolated_image = BilinearInterpolation(interpolation_size)([image, x])
    x = Conv2D(32, (3, 3), padding='same')(interpolated_image)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax', name='label')(x)
    return Model(image, [x, interpolated_image], name='STN')
