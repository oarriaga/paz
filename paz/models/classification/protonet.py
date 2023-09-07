import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Layer,
                                     ReLU, MaxPool2D, Flatten, Softmax)

from ...utils.documentation import docstring


def conv_block(x):
    """Basic convolution block used for prototypical networks.

    # Arguments
        x: Tensor.

    # Returns
        Tensor
    """
    x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2))(x)
    return x


def ProtoEmbedding(image_shape, num_blocks):
    """Embedding convolutional network used for proto-typical networks

    # Arguments:
        image_shape: List with image shape `(H, W, channels)`.
        num_blocks: Ints. Number of convolution blocks.

    # Returns:
        Keras model.

    # References:
        [prototypical networks](https://arxiv.org/abs/1703.05175)
    """
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
    """Compute euclidean distance for each vector x with each vector y

    # Arguments:
        x: Tensor with shape `(n, vector_dim)`
        y: Tensor with shape `(m, vector_dim)`

    # Returns:
        Tensor with shape `(n, m)` where each value pair n, m corresponds to
        the distance between the vector `n` of `x` with the vector `m` of `y`
    """
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)


@docstring(compute_pairwise_distances)
class ComputePairwiseDistances(Layer):
    def __init__(self, **kwargs):
        super(ComputePairwiseDistances, self).__init__(**kwargs)

    def call(self, z_queries, class_prototypes):
        return compute_pairwise_distances(z_queries, class_prototypes)


def ProtoNet(embed, num_classes, num_support, num_queries, image_shape):
    """Prototypical networks used for few-shot classification
    # Arguments:
        embed: Keras network for embedding images into metric space.
        num_classes: Number of `ways` for few-shot classification.
        num_support: Number of `shots` used for meta learning.
        num_queries: Number of test images to query.
        image_shape: List with image shape `(H, W, channels)`.

    # Returns:
        Keras model.

    # References:
        [prototypical networks](https://arxiv.org/abs/1703.05175)
    """
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
