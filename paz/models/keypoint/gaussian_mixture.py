from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras.layers import Concatenate


def block(x, filters, dilation_rate, alpha, kernel_size=(3, 3)):
    x = Conv2D(filters, kernel_size, dilation_rate=dilation_rate,
               padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    return x


def reduce(x, filters):
    x = Conv2D(filters, (5, 5), strides=(4, 4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def build_keypoint_map(x, keypoint_arg):
    x = Conv2D(4, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    components = x.shape[1] * x.shape[2]
    x = Reshape((1, components, 4), name="keypoint_%s" % keypoint_arg)(x)
    return x


def GaussianMixtureModel(input_shape, num_keypoints, filters=64, alpha=0.1):
    """2D probabilistic keypoints as per-keypoint Gaussian mixture maps.

    Each keypoint produces a ``(components, 4)`` map whose channels are the
    categorical logit, the scale, and the (x, y) mean offset of every mixture
    component. The mixture math lives in ``paz.backend.gaussian_mixture``.
    """
    image = inputs = Input(input_shape, name="image")
    for rate in [1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]:
        image = block(image, filters, (rate, rate), alpha)
    image = reduce(image, filters)
    image = reduce(image, filters)
    maps = [build_keypoint_map(image, arg) for arg in range(num_keypoints)]
    maps = Concatenate(axis=1, name="keypoints")(maps)
    return Model(inputs, maps, name="GaussianMixture")
