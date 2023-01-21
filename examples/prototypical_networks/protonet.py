import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Layer,
                                     ReLU, MaxPool2D, Flatten, Softmax)


def conv_block(x):
    x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2))(x)
    return x


def Embedding(image_shape, num_blocks):
    x = inputs = Input(image_shape)
    for _ in range(num_blocks):
        x = conv_block(x)
    z = Flatten()(x)
    return Model(inputs, z, name='EMBEDDING')


class FullReshape(Layer):
    """Reshapes all tensor dimensions including the batch dimension.
    """
    def __init__(self, shape, **kwargs):
        super(FullReshape, self).__init__(**kwargs)
        self.shape = shape

    def call(self, x):
        return tf.reshape(x, self.shape)


class ComputePrototypes(Layer):
    def __init__(self, axis=1, **kwargs):
        super(ComputePrototypes, self).__init__(**kwargs)
        self.axis = axis

    def call(self, z_support):
        class_prototypes = tf.reduce_mean(z_support, axis=self.axis)
        return class_prototypes


def compute_pairwise_distances(x, y):
    """Compute euclidean distance for each vector x with each vector y"""
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)


class ComputePairwiseDistances(Layer):
    def __init__(self, **kwargs):
        super(ComputePairwiseDistances, self).__init__(**kwargs)

    def call(self, z_queries, class_prototypes):
        return compute_pairwise_distances(z_queries, class_prototypes)


def PROTONET(embed, num_classes, num_support, num_queries, image_shape):
    support = Input((num_support, *image_shape), num_classes, name='support')
    queries = Input((num_queries, *image_shape), num_classes, name='queries')
    z_support = FullReshape((num_classes * num_support, *image_shape))(support)
    z_queries = FullReshape((num_classes * num_queries, *image_shape))(queries)
    z_support = embed(z_support)
    z_queries = embed(z_queries)
    z_dim = embed.output_shape[-1]
    z_support = FullReshape((num_classes, num_support, z_dim))(z_support)
    z_queries = FullReshape((num_classes * num_queries, z_dim))(z_queries)
    class_prototypes = ComputePrototypes(axis=1)(z_support)
    distances = ComputePairwiseDistances()(z_queries, class_prototypes)
    outputs = Softmax()(-distances)
    return Model(inputs=[support, queries], outputs=outputs, name='PROTONET')


def schedule(period=20, rate=0.5):
    def apply(epoch, learning_rate):
        if ((epoch % period) == 0) and (epoch != 0):
            learning_rate = rate * learning_rate
        return learning_rate
    return apply
