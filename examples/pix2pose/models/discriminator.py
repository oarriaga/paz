from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, BatchNormalization, LeakyReLU,
                                     Input, Flatten, Dense)


def convolution_block(x, filters):
    x = Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x


def Discriminator(input_shape=(128, 128, 3), name='PIX2POSE_DISCRIMINATOR'):
    input_image = Input(input_shape, name='input_image')
    x = convolution_block(input_image, 64)
    for filters in [128, 256, 512, 512, 512, 512]:
        x = convolution_block(x, filters)
    flatten = Flatten()(x)
    x = Dense(1, activation='sigmoid', name='discriminator_output')(flatten)
    model = Model(input_image, x, name=name)
    return model


model = Discriminator()
assert model.count_params() == 8640897
assert model.output_shape == (None, 1)
assert model.input_shape == (None, 128, 128, 3)
