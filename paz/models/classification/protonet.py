from keras import ops
import keras
from keras.layers import Input
from paz.layers import FullReshape, ComputePairwiseDistances, ComputePrototypes


def conv_block(x):
    """Basic convolution block used for prototypical networks.

    # Arguments
        x: Tensor.

    # Returns
        Tensor
    """
    x = keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool2D((2, 2))(x)
    return x


def ProtoEmbedding(input_shape, num_blocks):
    """Embedding convolutional network used for proto-typical networks

    # Arguments:
        input_shape: List with image shape `(H, W, channels)`.
        num_blocks: Ints. Number of convolution blocks.

    # Returns:
        Keras model.

    # References:
        [Prototypical Networks](https://arxiv.org/abs/1703.05175)
    """
    x = inputs = Input(input_shape)
    for _ in range(num_blocks):
        x = conv_block(x)
    z = keras.layers.Flatten()(x)
    return keras.Model(inputs, z, name="EMBEDDING")


def ProtoNet(embed, num_classes, num_support, num_queries, input_shape):
    """Prototypical networks used for few-shot classification

    # Arguments:
        embed: Keras network for embedding images into metric space.
        num_classes: Number of `ways` for few-shot classification.
        num_support: Number of `shots` used for meta learning.
        num_queries: Number of test images to query.
        input_shape: List with image shape `(H, W, channels)`.

    # Returns:
        Keras model.

    # References:
        [Prototypical Networks](https://arxiv.org/abs/1703.05175)
    """
    support = Input((num_support, *input_shape), num_classes, name="support")
    queries = Input((num_queries, *input_shape), num_classes, name="queries")
    z_support = FullReshape((num_classes * num_support, *input_shape))(support)
    z_queries = FullReshape((num_classes * num_queries, *input_shape))(queries)
    z_support = embed(z_support)
    z_queries = embed(z_queries)
    z_dim = embed.output_shape[-1]
    z_support = FullReshape((num_classes, num_support, z_dim))(z_support)
    z_queries = FullReshape((num_classes * num_queries, z_dim))(z_queries)
    class_prototypes = ComputePrototypes(axis=1)(z_support)
    distances = ComputePairwiseDistances()(z_queries, class_prototypes)
    outputs = keras.layers.Softmax()(-distances)
    return keras.Model([support, queries], outputs, name="PROTONET")
