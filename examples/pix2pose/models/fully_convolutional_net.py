from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Activation, LeakyReLU, BatchNormalization)


def block(x, filters, dilation_rate, alpha):
    x = Conv2D(filters, (3, 3), dilation_rate=dilation_rate, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    return x


def FullyConvolutionalNet(num_classes, input_shape, filters=64, alpha=0.1):
    """Fully convolutional network for segmentation.

    # Arguments
        num_classes: Int. Number of output channels.
        input_shape: List of integers indicating ``[H, W, num_channels]``.
        filters: Int. Number of filters used in convolutional layers.
        alpha: Float. Alpha parameter of leaky relu.

    # Returns
        Keras/tensorflow model

    # References
        - [Discovery of Latent 3D Keypoints via End-to-end
            Geometric Reasoning](https://arxiv.org/abs/1807.03146)
    """
    base = inputs = Input(input_shape, name='image')
    for base_arg, rate in enumerate([1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]):
        base = block(base, filters, (rate, rate), alpha)
    x = Conv2D(num_classes, (3, 3), padding='same')(base)
    outputs = Activation('softmax', name='masks')(x)
    model = Model(inputs, outputs, name='FULLY_CONVOLUTIONAL_NET')
    return model
