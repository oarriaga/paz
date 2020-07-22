import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Input
from tensorflow.keras.layers import BatchNormalization

from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

from paz.backend.keypoints import normalize_keypoints


def GaussianMixtureDistribution(X):
    batch_size, H, W, num_channels = X.shape
    categorical_values = K.reshape(X[:, :, :, 0], (batch_size, -1))
    variances = K.reshape(X[:, :, :, 1], (batch_size, -1))
    means_x = K.reshape(X[:, :, :, 2], (batch_size, -1, 1))
    means_y = K.reshape(X[:, :, :, 3], (batch_size, -1, 1))
    x_coordinates, y_coordinates = np.meshgrid(np.arange(H), np.arange(W))
    x_coordinates = np.reshape(x_coordinates, (-1, 1))
    y_coordinates = np.reshape(y_coordinates, (-1, 1))
    default_means = np.concatenate([x_coordinates, y_coordinates], axis=-1)
    default_means = normalize_keypoints(default_means, H, W)
    predicted_means = tf.concat([means_x, means_y], axis=-1)
    components = []
    for feature_map_arg, default_mean in enumerate(default_means):
        scale = K.cast(variances[:, feature_map_arg], K.floatx())
        scale = K.repeat_elements(K.expand_dims(scale, 1), 2, 1)
        default_mean = np.expand_dims(default_mean, axis=0)
        default_mean = np.repeat(default_mean, batch_size, 0)
        default_mean = default_mean.astype(np.float32)
        predicted_mean = predicted_means[:, feature_map_arg]
        mean = default_mean + predicted_mean
        component = tfd.MultivariateNormalDiag(mean, scale)
        components.append(component)
    categorical = tfd.Categorical(logits=categorical_values)
    gaussian_mixture = tfd.Mixture(categorical, components)
    return gaussian_mixture


def conv2D_block(x, filters, dilation_rate, alpha, kernel_size=(3, 3)):
    args = (filters, kernel_size)
    x = Conv2D(*args, dilation_rate=dilation_rate, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    return x


def GaussianMixtureModel(batch_shape, num_keypoints, filters=64, alpha=0.1):
    x = inputs = Input(batch_shape=batch_shape, name='image')
    for rate in [1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]:
        x = conv2D_block(x, filters, (rate, rate), alpha)
    x = Conv2D(filters, (5, 5), strides=(4, 4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, (5, 5), strides=(4, 4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    distributions = []
    for keypoint_arg in range(num_keypoints):
        k = Conv2D(4, (3, 3), strides=(1, 1), padding='same')(x)
        k = BatchNormalization()(k)
        k = LeakyReLU()(k)
        name = 'keypoint_%s' % keypoint_arg
        k = tfpl.DistributionLambda(GaussianMixtureDistribution, name=name)(k)
        distributions.append(k)
    return Model([inputs], distributions, name='GaussianMixture')
