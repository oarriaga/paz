
from pickle import FALSE
import numpy as np
import tensorflow as tf
from config import DETECTION_MODEL_PATH

from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.initializers import VarianceScaling 
from tensorflow.keras.initializers import truncated_normal 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


def zero_padding(inputs, pad_1, pad_2):
    pad_mat = np.array([[0, 0], [pad_1, pad_2], [pad_1, pad_2], [0, 0]])
    return tf.pad(tensor=inputs, paddings=pad_mat)


def conv_bn(inputs, oc, ks, st, scope, training, rate=1):
    if st == 1:
        layer = Conv2D(oc, ks, strides=st, padding='SAME', use_bias=False,
                        dilation_rate=rate,
                        kernel_regularizer=l2(0.5 * (1.0)),
                        name=scope + '/conv2d',
                        kernel_initializer=VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    )(inputs)
    else:
        pad_total = ks - 1
        pad_1 = pad_total // 2
        pad_2 = pad_total - pad_1
        padded_inputs = zero_padding(inputs, pad_1, pad_2)
        layer = Conv2D(
            oc, ks, strides=st, padding='VALID', use_bias=False,
            dilation_rate=rate,
            name=scope+ '/conv2d',
            kernel_regularizer=l2(0.5 * (1.0)),
            kernel_initializer=VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )(padded_inputs)
    layer = BatchNormalization(name=scope+'/batch_normalization')(layer, training=training)
    return layer



def conv_bn_relu(inputs, oc, ks, st, scope, training, rate=1):
    layer = conv_bn(inputs, oc, ks, st, scope, training, rate=rate)
    layer = ReLU()(layer)
    return layer


def bottleneck(inputs, oc, st, scope, training, rate=1):
    ic = inputs.get_shape().as_list()[-1]
    if ic == oc:
        if st == 1:
            shortcut = inputs
        else:
            shortcut = MaxPool2D((st, st), (st, st), 'SAME')(inputs)
    else:
        shortcut = conv_bn(inputs, oc, 1, st, scope + '/shortcut', training)

    residual = conv_bn_relu(inputs, oc//4, 1, 1, scope + '/conv1', training)
    residual = conv_bn_relu(residual, oc//4, 3, st, scope + '/conv2', training, rate)
    residual = conv_bn(residual, oc, 1, 1, scope + '/conv3', training)
    output = ReLU()(shortcut + residual)

    return output


def resnet50(inputs, scope, training):
    layer = conv_bn_relu(inputs, 64, 7, 2, scope + '/conv1', training)

    for unit in range(2):
        layer = bottleneck(layer, 256, 1, scope + '/block1/unit%d' % (unit+1), training)
    layer = bottleneck(layer, 256, 2, scope + '/block1/unit3', training)

    for unit in range(4):
        layer = bottleneck(layer, 512, 1, scope + '/block2/unit%d' % (unit+1), training, 2)

    for unit in range(6):
        layer = bottleneck(layer, 1024, 1, scope + '/block3/unit%d' % (unit+1), training, 4)

    layer = conv_bn_relu(layer, 256, 3, 1, scope + '/squeeze', training)

    return layer


def net_2d(features, training, scope, n_out):
    layer = conv_bn_relu(features, 256, 3, 1, scope + '/project', training)
    hmap = Conv2D(
    n_out, 1, strides=1, padding='SAME',
    activation=tf.nn.sigmoid,
    name=scope + '/prediction/conv2d',
    kernel_initializer=truncated_normal(stddev=0.01)
    )(layer)
    return hmap


def net_3d(features, training, scope, n_out, need_norm):
    layer = conv_bn_relu(features, 256, 3, 1, scope + '/project', training)
    dmap_raw = Conv2D(
    n_out * 3, 1, strides=1, padding='SAME',
    activation=None,
    name=scope + '/prediction/conv2d',
    kernel_initializer=truncated_normal(stddev=0.01)
    )(layer)
    if need_norm:
        dmap_norm = tf.norm(tensor=dmap_raw, axis=-1, keepdims=True)
        dmap = dmap_raw / tf.maximum(dmap_norm, 1e-6)
    else:
        dmap = dmap_raw

    h, w = features.get_shape().as_list()[1:3]
    dmap = tf.reshape(dmap, [-1, h, w, n_out, 3])

    if need_norm:
        return dmap, dmap_norm

    return dmap


def get_pose_tile(N):
    pos_tile = tf.tile(
        tf.constant(
        np.expand_dims(
            np.stack(
            [
                np.tile(np.linspace(-1, 1, 32).reshape([1, 32]), [32, 1]),
                np.tile(np.linspace(-1, 1, 32).reshape([32, 1]), [1, 32])
            ], -1
            ), 0
        ), dtype=tf.float32
        ), [N, 1, 1, 1]
    )
    return pos_tile


def detnet(img, n_stack, scope, training):
    features = resnet50(img, scope + '/resnet', training)
    pos_tile = get_pose_tile(tf.shape(input=img)[0])
    features = tf.concat([features, pos_tile], -1)

    hmaps = []
    dmaps = []
    lmaps = []
    for i in range(n_stack):
        hmap = net_2d(features, training, scope + '/hmap_%d' % i, 21)
        features = tf.concat([features, hmap], axis=-1)
        hmaps.append(hmap)

        dmap = net_3d(features, training, scope + '/dmap_%d' % i, 21, False)
        features = tf.concat([features, tf.reshape(dmap, [-1, 32, 32, 21 * 3])], -1)
        dmaps.append(dmap)

        lmap = net_3d(features, training, scope + '/lmap_%d' % i, 21, False)
        features = tf.concat([features, tf.reshape(lmap, [-1, 32, 32, 21 * 3])], -1)
        lmaps.append(lmap)

    return hmaps, dmaps, lmaps


def tf_hmap_to_uv(hmap):
    hmap_flat = tf.reshape(hmap, (tf.shape(input=hmap)[0], -1, tf.shape(input=hmap)[3]))
    argmax = tf.argmax(input=hmap_flat, axis=1, output_type=tf.int32)
    argmax_x = argmax // tf.shape(input=hmap)[2]
    argmax_y = argmax % tf.shape(input=hmap)[2]
    uv = tf.stack((argmax_x, argmax_y), axis=1)
    uv = tf.transpose(a=uv, perm=[0, 2, 1])
    return uv


class DetNet:
    """
    DetNet: estimating 3D keypoint positions from input color image.
    """

    def __init__(self):
        """
    Parameters
    ----------
    model_path : str
      Path to the trained model.
    """
        model_path = DETECTION_MODEL_PATH
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=config)
            self.input_ph = tf.keras.Input(shape=(128, 128, 3), dtype=tf.uint8)
            self.input_ph = tf.squeeze(self.input_ph, 0)
            self.feed_img = \
                tf.cast(tf.expand_dims(self.input_ph, 0), tf.float32) / 255
            self.hmaps, self.dmaps, self.lmaps = \
                detnet(self.feed_img, 1, 'prior_based_hand', False)

            self.hmap = self.hmaps[-1]
            self.dmap = self.dmaps[-1]
            self.lmap = self.lmaps[-1]

            self.uv = tf_hmap_to_uv(self.hmap)
            self.delta = tf.gather_nd(
                tf.transpose(a=self.dmap, perm=[0, 3, 1, 2, 4]), self.uv, batch_dims=2
            )[0]
            self.xyz = tf.gather_nd(
                tf.transpose(a=self.lmap, perm=[0, 3, 1, 2, 4]), self.uv, batch_dims=2
            )[0]

            self.uv = self.uv[0]
            tf.compat.v1.train.Saver().restore(self.sess, model_path)
            # print(tf.train.list_variables(model_path))


    def process(self, img):
        """
    Process a color image.

    Parameters
    ----------
    img : np.ndarray
      A 128x128 RGB image of **left hand** with dtype uint8.

    Returns
    -------
    np.ndarray, shape [21, 3]
      Normalized keypoint locations. The coordinates are relative to the M0
      joint and normalized by the length of the bone from wrist to M0. The
      order of keypoints is as `kinematics.MPIIHandJoints`.
    np.ndarray, shape [21, 2]
      The uv coordinates of the keypoints on the heat map, whose resolution is
      32x32.
    """
        results = self.sess.run([self.xyz, self.uv], {self.input_ph: img})
        return results
