
import os
import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import truncated_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Activation, Reshape
from tensorflow.keras.models import Model


WEIGHT_PATH = ('https://github.com/oarriaga/altamira-data/releases/download/'
               'v0.14/iknet_weight.hdf5')


def dense(x, num_units):
    x = Dense(num_units, activation=None, kernel_regularizer=l2(0.5 * 1.0),
              kernel_initializer=truncated_normal(stddev=0.01))(x)
    return x


def block(x, num_units):
    x = dense(x, num_units)
    x = BatchNormalization()(x)
    return x


def normalize(x):
    norm = tf.norm(x, axis=-1, keepdims=True)
    norm = tf.maximum(norm, 1e-6)
    normalized_x = x / norm
    return normalized_x


def reorder_quaternions(quaternions):
    w = quaternions[:, :, 0:1]
    qs = quaternions[:, :, 1:4]
    quaternions = tf.concat((qs, w), axis=-1)
    return quaternions


def IKNet(input_shape=(84, 3), num_keypoints=21, depth=6, width=1024):
    """IKNet: Estimate absolute joint angle for the minimal hand keypoints.

    # Arguments
        input_shape: [num_keypoint x 4, 3]. Contains 3D keypoints, bone
                     orientation, refrence keypoint, refrence bone orientation.
        num_keypoints: Int. Number of keypoints.

    # Returns
        Tensorflow-Keras model.
        absolute joint angle in quaternion representation.

    # Reference
        - [Monocular Real-time Hand Shape and Motion Capture using Multi-modal
           Data](https://arxiv.org/abs/2003.09572)
    """
    input = Input(shape=input_shape, dtype=tf.float32)
    x = Reshape([1, -1])(input)

    for depth_arg in range(depth):
        x = block(x, width)
        x = Activation('sigmoid')(x)
    x = dense(x, num_keypoints * 4)
    x = Reshape([num_keypoints, 4])(x)
    x = normalize(x)

    positive_mask = tf.tile(x[:, :, 0:1] > 0, [1, 1, 4])
    quaternions = tf.where(positive_mask, x, -x)
    quaternions = reorder_quaternions(quaternions)

    model = Model(input, outputs=[quaternions])

    URL = ('https://github.com/oarriaga/altamira-data/releases/download/'
           'v0.14/iknet_weight.hdf5')
    filename = os.path.basename(URL)
    weights_path = get_file(filename, URL, cache_subdir='paz/models')
    print('==> Loading %s model weights' % weights_path)
    model.load_weights(weights_path)
    return model
