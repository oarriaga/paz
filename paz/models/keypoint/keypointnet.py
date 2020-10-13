from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model

from ..layers import ExpectedDepth
from ..layers import ExpectedValue2D
from ..layers import SubtractScalar


def block(x, num_filters, dilation_rate, alpha, name, kernel_size=(3, 3)):
    x = Conv2D(num_filters, kernel_size, dilation_rate=dilation_rate,
               padding='same', name=name)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    return x


def KeypointNet2D(input_shape, num_keypoints, filters=64, alpha=0.1):
    """Model for discovering keypoint locations in 2D space, modified from

    # Arguments
        input_shape: List of integers indicating ``[H, W, num_channels]``.
        num_keypoints: Int. Number of keypoints to discover.
        filters: Int. Number of filters used in convolutional layers.
        alpha: Float. Alpha parameter of leaky relu.

    # Returns
        Keras/tensorflow model

    # References
        - [Discovery of Latent 3D Keypoints via End-to-end
            Geometric Reasoning](https://arxiv.org/abs/1807.03146)
    """
    width, height = input_shape[:2]
    base = input_tensor = Input(input_shape, name='image')
    for base_arg, rate in enumerate([1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]):
        name = 'conv2D_base-%s' % base_arg
        base = block(base, filters, (rate, rate), alpha, name)

    name = 'uv_volume_features-%s'
    uv_volume = Conv2D(num_keypoints, (3, 3),
                       padding='same', name=name % 0)(base)
    uv_volume = Permute([3, 1, 2], name=name % 1)(uv_volume)
    volume_shape = [num_keypoints, width * height]
    uv_volume = Reshape(volume_shape, name=name % 2)(uv_volume)
    uv_volume = Activation('softmax', name=name % 3)(uv_volume)
    volume_shape = [num_keypoints, width, height]
    uv_volume = Reshape(volume_shape, name='uv_volume')(uv_volume)
    uv = ExpectedValue2D(name='keypoints')(uv_volume)
    model = Model(input_tensor, uv, name='keypointnet2D')
    return model


def KeypointNet(input_shape, num_keypoints, depth=.2, filters=64, alpha=0.1):
    """Keypointnet model for discovering keypoint locations in 3D space

    # Arguments
        input_shape: List of integers indicating ``[H, W, num_channels)``.
        num_keypoints: Int. Number of keypoints to discover.
        depth: Float. Prior depth (centimeters) of keypoints.
        filters: Int. Number of filters used in convolutional layers.
        alpha: Float. Alpha parameter of leaky relu.

    # Returns
        Keras/tensorflow model

    # References
        - [Discovery of Latent 3D Keypoints via End-to-end
            Geometric Reasoning](https://arxiv.org/abs/1807.03146)
    """
    width, height = input_shape[:2]
    base = input_tensor = Input(input_shape, name='image')
    for base_arg, rate in enumerate([1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]):
        name = 'conv2D_base-%s' % base_arg
        base = block(base, filters, (rate, rate), alpha, name)

    name = 'uv_volume_features-%s'
    uv_volume = Conv2D(num_keypoints, (3, 3),
                       padding='same', name=name % 0)(base)
    uv_volume = Permute([3, 1, 2], name=name % 1)(uv_volume)
    volume_shape = [num_keypoints, width * height]
    uv_volume = Reshape(volume_shape, name=name % 2)(uv_volume)
    uv_volume = Activation('softmax', name=name % 3)(uv_volume)
    volume_shape = [num_keypoints, width, height]
    uv_volume = Reshape(volume_shape, name='uv_volume')(uv_volume)
    uv = ExpectedValue2D(name='expected_uv')(uv_volume)

    name = 'depth_volume_features-%s'
    depth_volume = Conv2D(num_keypoints, (3, 3),
                          padding='same', name=name % 0)(base)
    depth_volume = SubtractScalar(depth, name=name % 1)(depth_volume)
    depth_volume = Permute([3, 1, 2], name='depth_volume')(depth_volume)
    z = ExpectedDepth(name='expected_z')([depth_volume, uv_volume])
    uvz = Concatenate(axis=-1, name='uvz_points')([uv, z])
    model = Model(input_tensor, [uvz, uv_volume], name='keypointnet')
    return model


def KeypointNetShared(input_shape, num_keypoints, depth, filters, alpha):
    """Keypointnet shared model with two views as input.

    # Arguments
        input_shape: List of integers indicating ``[H, W, num_channels]``.
        num_keypoints: Int. Number of keypoints to discover.
        depth: Float. Prior depth (centimeters) of keypoints.
        filters: Int. Number of filters used in convolutional layers.
        alpha: Float. Alpha parameter of leaky relu.

    # Returns
        Keras/tensorflow model

    # References
        - [Discovery of Latent 3D Keypoints via End-to-end
            Geometric Reasoning](https://arxiv.org/abs/1807.03146)
    """

    model_args = (input_shape, num_keypoints, depth, filters, alpha)
    keypointnet = KeypointNet(*model_args)
    image_A = Input(input_shape, name='image_A')
    image_B = Input(input_shape, name='image_B')
    uvz_A, uv_volume_A = keypointnet(image_A)
    uvz_B, uv_volume_B = keypointnet(image_B)
    uvz_points = Concatenate(axis=1, name='uvz_points-shared')([uvz_A, uvz_B])
    uv_volumes = Concatenate(axis=1, name='uv_volumes-shared')(
        [uv_volume_A, uv_volume_B])
    inputs, outputs = [image_A, image_B], [uvz_points, uv_volumes]
    return Model(inputs, outputs, name='keypointnet-shared')
