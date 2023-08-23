from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Flatten, BatchNormalization,
    ReLU, Conv2D, MaxPool2D, Layer, Softmax)

from paz.datasets.omniglot import load, sample_between_alphabet


def conv_block(x):
    x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2))(x)
    return x


def CONVNET(num_classes, image_shape, num_blocks=4):
    x = inputs = Input(image_shape)
    for _ in range(num_blocks):
        x = conv_block(x)
    x = Flatten()(x)
    outputs = Dense(num_classes)(x)
    return Model(inputs, outputs, name='CONVNET')


class MAML(Model):
    def train_step(self, data):
        print(data)


seed = 777
train_ways = 5
train_shots = 10
train_queries = 10
train_args = (train_ways, train_shots, train_queries)
image_shape = (28, 28, 1)
RNG = np.random.default_rng(seed)
train_data = load('train', image_shape[:2], True)
train_sampler = partial(sample_between_alphabet, RNG, train_data, *train_args)
(x1, y1), (x2, y2) = train_sampler()


class FullReshape(Layer):
    """Reshapes all tensor dimensions including the batch dimension.
    """
    def __init__(self, shape, **kwargs):
        super(FullReshape, self).__init__(**kwargs)
        self.shape = shape

    def call(self, x):
        return tf.reshape(x, self.shape)


def MAML(embed, num_classes, num_support, num_queries, image_shape):
    support = Input((num_support, *image_shape), num_classes, name='support')
    queries = Input((num_queries, *image_shape), num_classes, name='queries')
    z_support = FullReshape((num_classes * num_support, *image_shape))(support)
    z_queries = FullReshape((num_classes * num_queries, *image_shape))(queries)
    logits_support = embed(z_support)
    logits_queries = embed(z_queries)
    y_support = Softmax(logits_support, name='y_support')
    y_queries = Softmax(logits_queries, name='y_support')
    return Model(inputs=[support, queries],
                 outputs=[y_support, y_queries], name='MAML')


