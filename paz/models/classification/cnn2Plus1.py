import tensorflow as tf

keras = tf.keras
from keras import Sequential
from keras.models import Model
from keras.layers import (Conv3D, Input, BatchNormalization, Layer, GlobalAveragePooling3D, ReLU, Flatten,
                          LayerNormalization, Dense, add)
import einops
import random

from ..layers import Conv2DNormalization

"""
References:
    - [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248v3)
    - [Video classification with a 3D convolutional neural network](https://www.tensorflow.org/tutorials/video/video_classification#load_and_preprocess_video_data)
"""


class Conv2Plus1D(Layer):
    def __init__(self, filters, kernel_size, padding):
        """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension.
    """
        super().__init__()
        initializer_glorot_spatial = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
        initializer_glorot_temporal = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))

        self.seq = Sequential([
            # Spatial decomposition
            Conv3D(filters=filters,
                   kernel_size=(1, kernel_size[1], kernel_size[2]),
                   padding=padding,
                   kernel_initializer=initializer_glorot_spatial),
            # Temporal decomposition
            Conv3D(filters=filters,
                   kernel_size=(kernel_size[0], 1, 1),
                   padding=padding,
                   kernel_initializer=initializer_glorot_temporal)
        ])

    def call(self, x):
        return self.seq(x)


class ResidualMain(Layer):
    """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = Sequential([
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            LayerNormalization(),
            ReLU(),
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)


class Project(Layer):
    """
    Project certain dimensions of the tensor as the data is passed through different
    sized filters and downsampled.
  """

    def __init__(self, units):
        super().__init__()
        initializer_glorot = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))

        self.seq = Sequential([
            Dense(units, kernel_initializer=initializer_glorot),
            LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)


def add_residual_block(input, filters, kernel_size):
    """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
    out = ResidualMain(filters, kernel_size)(input)

    res = input
    # Using the Keras functional APIs, project the last dimension of the tensor to
    # match the new filter size
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return add([res, out])


class ResizeVideo(Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = tf.keras.layers.Resizing(self.height, self.width)  # care replaced Resized with reshaped

    @tf.function
    def call(self, video):
        """
      Use the einops library to resize the tensor.

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
        # b stands for batch size, t stands for time, h stands for height,
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t=old_shape['t'])
        return videos


def CNN2Plus1D(weights=None, input_shape=(38, 96, 96, 3), seed=305865):
    """Binary Classification for videos with 2+1D CNNs.
    # Arguments
        weights: String, path to the weights file to load. TODO add weights implementation when weights are available
        input_shape: List of integers. Input shape to the model in following format: (frames, height, width, channels)
        e.g. (38, 96, 96, 3).

    # Reference
        - [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248v3)
        - [Video classification with a 3D convolutional neural network]
        (https://www.tensorflow.org/tutorials/video/video_classification#load_and_preprocess_video_data)
    """
    if len(input_shape) != 4:
        raise ValueError(
            '`input_shape` must be a tuple of 4 integers. '
            'Received: %s' % (input_shape,))

    random.seed(seed)
    initializer_glorot_output = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))

    HEIGHT = input_shape[1]
    WIDTH = input_shape[2]

    # input_shape = (None, 10, HEIGHT, WIDTH, 3)
    image = Input(shape=input_shape, name='image')
    x = image

    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    x = GlobalAveragePooling3D()(x)
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid", kernel_initializer=initializer_glorot_output)(x)

    model = Model(inputs=image, outputs=x, name='Vvad2Plus1D')

    return model
