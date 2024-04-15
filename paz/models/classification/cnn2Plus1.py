import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv3D, Input, BatchNormalization, Layer, GlobalAveragePooling3D, ReLU, Flatten,
                                     LayerNormalization, Dense, add)
from tensorflow.keras.utils import get_file
from tensorflow.keras.initializers import GlorotUniform


Architecture_Options = ['CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers', 'CNN2Plus1D_Light',
                        'CNN2Plus1D_18']
URL = 'https://github.com/oarriaga/altamira-data/releases/download/v0.19/'


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
        initializer_glorot = GlorotUniform(seed=random.randint(0, 1_000_000))

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

    residual = input

    if out.shape[-1] != input.shape[-1]:
        residual = Project(out.shape[-1])(residual)

    return add([residual, out])


class ResizeVideo(Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = tf.keras.layers.Resizing(self.height, self.width)  # care replaced Resized with reshaped

    @tf.function
    def call(self, video):
        """Use tensorflow reshape to resize the tensor.

          # Arguments
            video: Tensor representation of the video, in the form of a set of frames.

          # Return
            A downsampled size of the video according to the new height and width it should be resized to.
        """
        video_shape = tf.shape(video)
        video_reshaped = tf.reshape(video, [-1, video_shape[2], video_shape[3], video_shape[4]])

        images_resized = self.resizing_layer(video_reshaped)

        videos = tf.reshape(images_resized,
                            [video_shape[0], video_shape[1], self.height, self.width, video_shape[4]])
        return videos


def normal(input_layer, height, width):
    """ CNN2+1D architecture is based on the tensorflow implementation
        # Arguments
            input_layer: Tensorflow input layer of the network
            height: Height of the input video
            width: Width of the input video
    """
    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(height // 4, width // 4)(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(height // 8, width // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(height // 16, width // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))
    return x


def filters(input_layer, height, width):
    """ Architecture is a CNN2+1D version with increased filter sizes
        # Arguments
            input_layer: Tensorflow input layer of the network
            height: Height of the input video
            width: Width of the input video
    """

    x = Conv2Plus1D(filters=32, kernel_size=(3, 7, 7), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    # Block 1
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(height // 4, width // 4)(x)

    # Block 2
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(height // 8, width // 8)(x)

    # Block 3
    x = add_residual_block(x, 128, (3, 3, 3))
    x = ResizeVideo(height // 16, width // 16)(x)

    # Block 4
    x = add_residual_block(x, 256, (3, 3, 3))
    return x


def layers(input_layer, height, width):
    """ Architecture is a CNN2+1D version with doubled layers
        # Arguments
            input_layer: Tensorflow input layer of the network
            height: Height of the input video
            width: Width of the input video
    """
    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(height // 4, width // 4)(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(height // 8, width // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(height // 16, width // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))
    x = add_residual_block(x, 128, (3, 3, 3))
    return x


def light(input_layer, height, width):
    """ Architecture is a CNN2+1D version with one layer less
        # Arguments
            input_layer: Tensorflow input layer of the network
            height: Height of the input video
            width: Width of the input video
    """
    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(height // 4, width // 4)(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(height // 8, width // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    return x


def original_18(input_layer, height, width):
    """ Architecture is a CNN2+1D version based on the original paper.
        # Arguments
            input_layer: Tensorflow input layer of the network
            height: Height of the input video
            width: Width of the input video
    """
    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    # Block 1
    x = add_residual_block(x, 64, (3, 3, 3))
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(height // 4, width // 4)(x)

    # Block 2
    x = add_residual_block(x, 128, (3, 3, 3))
    x = add_residual_block(x, 128, (3, 3, 3))
    x = ResizeVideo(height // 8, width // 8)(x)

    # Block 3
    x = add_residual_block(x, 256, (3, 3, 3))
    x = add_residual_block(x, 256, (3, 3, 3))
    x = ResizeVideo(height // 16, width // 16)(x)

    # Block 4
    x = add_residual_block(x, 512, (3, 3, 3))
    x = add_residual_block(x, 512, (3, 3, 3))
    return x


def CNN2Plus1D(weights=None, input_shape=(38, 96, 96, 3), seed=305865,
               architecture='CNN2Plus1D'):
    """Binary Classification for videos with 2+1D CNNs.
    # Arguments
        weights: ``None`` or string with pre-trained dataset. Valid datasets
            include only ``VVAD-LRS3``.
        input_shape: List of integers. Input shape to the model in following format: (frames, height, width, channels)
        e.g. (38, 96, 96, 3).
        seed: Integer. Seed for random number generator.
        architecture: String. Name of the architecture to use. Currently supported: 'CNN2Plus1D', 'CNN2Plus1D_Filters',
            'CNN2Plus1D_Layers', 'CNN2Plus1D_Light'. 'CNN2Plus1D_18' is only available without weights.

    # Reference
        - [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248v3)
        - [Video classification with a 3D convolutional neural network]
        (https://www.tensorflow.org/tutorials/video/video_classification#load_and_preprocess_video_data)
    """
    if len(input_shape) != 4:
        raise ValueError(
            '`input_shape` must be a tuple of 4 integers. '
            'Received: %s' % (input_shape,))
    assert architecture in Architecture_Options, f"'{architecture}' is not in {Architecture_Options}"

    random.seed(seed)
    initializer_glorot_output = GlorotUniform(seed=random.randint(0, 1_000_000))

    # input_shape = (None, 10, HEIGHT, WIDTH, 3)
    image = Input(shape=input_shape, name='image')
    x = image

    weights_path = ''
    if architecture == 'CNN2Plus1D':
        x = normal(x, input_shape[1], input_shape[2])
        if weights == 'VVAD_LRS3':
            filename = 'cnn-2plus1d_weights-21.hdf5'
            weights_path = get_file(filename, URL + filename, cache_subdir='paz/models')
    elif architecture == 'CNN2Plus1D_Filters':
        x = filters(x, input_shape[1], input_shape[2])
        if weights == 'VVAD_LRS3':
            filename = 'cnn-2plus1d-filters_weights-21.hdf5'
            weights_path = get_file(filename, URL + filename, cache_subdir='paz/models')
    elif architecture == 'CNN2Plus1D_Layers':
        x = layers(x, input_shape[1], input_shape[2])
        if weights == 'VVAD_LRS3':
            filename = 'cnn-2plus1d-layers_weights-17.hdf5'
            weights_path = get_file(filename, URL + filename, cache_subdir='paz/models')
    elif architecture == 'CNN2Plus1D_Light':
        x = light(x, input_shape[1], input_shape[2])
        if weights == 'VVAD_LRS3':
            filename = 'cnn-2plus1d-light_weights-35.hdf5'
            weights_path = get_file(filename, URL + filename, cache_subdir='paz/models')
    elif architecture == 'CNN2Plus1D_18':
        x = original_18(x, input_shape[1], input_shape[2])

    x = GlobalAveragePooling3D()(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid', kernel_initializer=initializer_glorot_output)(x)

    model = Model(inputs=image, outputs=x, name=architecture)

    if weights == 'VVAD_LRS3':
        if architecture == 'CNN2Plus1D_18':
            raise ValueError(f"'{architecture}' is not available with weights.")
        model.load_weights(weights_path)

    return model
