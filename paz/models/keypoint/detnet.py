
import numpy as np
import sys
import os
import keras
import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.initializers import truncated_normal
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


WEIGHT_PATH = ('https://github.com/oarriaga/altamira-data/releases/download'
               '_v0.14/detnet_weights.hdf5')


class ZeroPadding(Layer):
    def __init__(self, pad_1, pad_2, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_1 = pad_1
        self.pad_2 = pad_2

    def zero_padding(self, inputs):
        pad_mat = np.array([[0, 0],
                            [self.pad_1, self.pad_2],
                            [self.pad_1, self.pad_2],
                            [0, 0]])
        return tf.pad(inputs, paddings=pad_mat)

    def call(self, inputs):
        outputs = self.zero_padding(inputs)
        return outputs


class GatherND(Layer):
    def __init__(self, uv, batch_dims=0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.batch_dims = batch_dims
        self.uv = uv
        self.num_keypoints = 21

    def compute_output_shape(self):
        return (None, 21, 3)

    def gather_nd(self, inputs):
        return tf.gather_nd(inputs, self.uv, batch_dims=self.batch_dims)

    def call(self, inputs):
        outputs = self.gather_nd(inputs)
        return outputs


def block(tensor, filters, kernel_size, strides, name, rate=1, with_relu=True):
    if strides == 1:
        x = Conv2D(filters, kernel_size, strides, padding='SAME',
                   use_bias=False, dilation_rate=rate,
                   kernel_regularizer=l2(0.5 * 1.0), name=name + '_conv2d',
                   kernel_initializer=VarianceScaling(
                       mode="fan_avg", distribution="uniform"))(tensor)
    else:
        pad_1 = (kernel_size - 1) // 2
        pad_2 = (kernel_size - 1) - pad_1
        x = ZeroPadding(pad_1, pad_2)(tensor)
        x = Conv2D(filters, kernel_size, strides, padding='VALID',
                   use_bias=False, dilation_rate=rate,
                   kernel_regularizer=l2(0.5 * (1.0)), name=name + '_conv2d',
                   kernel_initializer=VarianceScaling(
                       mode="fan_avg", distribution="uniform"))(x)
    x = BatchNormalization(name=name + '_batch_normalization')(x)
    if with_relu:
        x = ReLU()(x)
    return x


def bottleneck(tensor, filters, strides, name, rate=1):
    shape = K.int_shape(tensor)[-1]
    if shape == filters:
        if strides == 1:
            x = tensor
        else:
            x = MaxPool2D(strides, strides, 'same')(tensor)
    else:
        x = block(tensor, filters, 1, strides, name + '_shortcut',
                  with_relu=False)
    residual = block(tensor, (filters // 4), 1, 1, name + '_conv1')
    residual = block(residual, (filters // 4), 3, strides, name + '_conv2',
                     rate)
    residual = block(residual, filters, 1, 1, name + '_conv3',
                     with_relu=False)
    output = ReLU()(x + residual)
    return output


def resnet50(tensor, name):
    x = block(tensor, 64, 7, 2, name + '_conv1')
    for arg in range(2):
        x = bottleneck(x, 256, 1, name + '_block1_unit%d' % (arg + 1))
    x = bottleneck(x, 256, 2, name + '_block1_unit3')
    for arg in range(4):
        x = bottleneck(x, 512, 1, name + '_block2_unit%d' % (arg + 1), 2)
    for arg in range(6):
        x = bottleneck(x, 1024, 1, name + '_block3_unit%d' % (arg + 1), 4)
    x = block(x, 256, 3, 1, name + '_squeeze')
    return x


def net2D(features, num_keypoints, name):
    x = block(features, 256, 3, 1, name + '_project')
    heat_map = Conv2D(num_keypoints, 1, strides=1, padding='SAME',
                      activation='sigmoid', name=name + '_prediction_conv2d',
                      kernel_initializer=truncated_normal(stddev=0.01))(x)
    return heat_map


def net3D(features, num_keypoints, name, need_norm=False):
    x = block(features, 256, 3, 1, name + '_project')
    delta_map = Conv2D(num_keypoints * 3, 1, strides=1, padding='SAME',
                       name=name + '_prediction_conv2d',
                       kernel_initializer=truncated_normal(stddev=0.01))(x)
    if need_norm:
        delta_map_norm = tf.norm(delta_map, axis=-1, keepdims=True)
        delta_map = delta_map / tf.maximum(delta_map_norm, 1e-6)

    H, W = K.int_shape(features)[1:3]
    delta_map = Reshape([H, W, num_keypoints, 3])(delta_map)
    if need_norm:
        return delta_map, delta_map_norm
    return delta_map


def get_pose_tile(N):
    x = np.linspace(-1, 1, 32)
    x = np.stack([np.tile(x.reshape([1, 32]), [32, 1]),
                  np.tile(x.reshape([32, 1]), [1, 32])], -1)
    x = np.expand_dims(x, 0)
    x = tf.constant(x, dtype=tf.float32)
    pose_tile = tf.tile(x, [1, 1, 1, 1])
    return pose_tile


def tf_heatmap_to_uv(heatmap):
    shape = K.int_shape(heatmap)
    heatmap = Reshape((-1, shape[3]))(heatmap)
    argmax = keras.ops.argmax(heatmap, axis=1)
    argmax_x = argmax // shape[2]
    argmax_y = argmax % shape[2]
    uv = keras.ops.stack((argmax_x, argmax_y), axis=1)
    uv = keras.ops.transpose(uv, [0, 2, 1])
    return uv


def DetNet(input_shape=(128, 128, 3), num_keypoints=21):
    """DetNet: Estimate 3D keypoint positions of minimal hand from input
               color image.

    # Arguments
        input_shape: Shape for 128x128 RGB image of **left hand**.
                     List of integers. Input shape to the model including only
                     spatial and channel resolution e.g. (128, 128, 3).
        num_keypoints: Int. Number of keypoints.

    # Returns
        Tensorflow-Keras model.
        xyz: Numpy array [num_keypoints, 3]. Normalized 3D keypoint locations.
        uv: Numpy array [num_keypoints, 2]. The uv coordinates of the keypoints
            on the heat map, whose resolution is 32x32.

    # Reference
        -[Monocular Real-time Hand Shape and Motion Capture using Multi-modal
          Data](https://arxiv.org/abs/2003.09572)
    """

    image = Input(shape=input_shape, dtype=tf.uint8)
    x = keras.ops.cast(image, tf.float32) / 255

    name = 'prior_based_hand'
    features = resnet50(x, name + '_resnet')
    pose_tile = get_pose_tile(K.int_shape(x)[0])

    features = concatenate([features, pose_tile], -1)
    heat_map = net2D(features, num_keypoints, name + '_hmap_0')
    features = concatenate([features, heat_map], axis=-1)

    delta_map = net3D(features, num_keypoints, name + '_dmap_0')
    delta_map_reshaped = Reshape([32, 32, num_keypoints * 3])(delta_map)
    features = concatenate([features, delta_map_reshaped], -1)

    location_map = net3D(features, num_keypoints, name + '_lmap_0')
    location_map_reshaped = Reshape([32, 32, num_keypoints * 3])(location_map)
    features = concatenate([features, location_map_reshaped], -1)

    uv = tf_heatmap_to_uv(heat_map)

    xyz = keras.ops.transpose(location_map, [0, 3, 1, 2, 4])
    # xyz = keras.ops.take(xyz[0], uv[0], axis=0)
    # print(K.int_shape(xyz))
    xyz = GatherND(uv, batch_dims=2)(xyz)

    xyz = xyz[0]
    uv = uv[0]
    model = Model(image, outputs=[xyz, uv])
    # print(model.summary())

    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '_v0.14/detnet_weights.hdf5')
    filename = os.path.basename(URL)
    weights_path = get_file(filename, URL, cache_subdir='paz/models')
    print('==> Loading %s model weights' % weights_path)
    model.load_weights(weights_path)
    return model
