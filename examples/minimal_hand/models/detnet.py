
import numpy as np
import tensorflow as tf
from config import DETECTION_MODEL_PATH


def zero_padding(inputs, pad_1, pad_2):
    pad_mat = np.array([[0, 0], [pad_1, pad_2], [pad_1, pad_2], [0, 0]])
    return tf.pad(tensor=inputs, paddings=pad_mat)


def conv_bn(inputs, oc, ks, st, scope, training, rate=1):
    with tf.compat.v1.variable_scope(scope):
        if st == 1:
            layer = tf.compat.v1.layers.conv2d(
            inputs, oc, ks, strides=st, padding='SAME', use_bias=False,
            dilation_rate=rate,
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1.0)),
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        )
        else:
            pad_total = ks - 1
            pad_1 = pad_total // 2
            pad_2 = pad_total - pad_1
            padded_inputs = zero_padding(inputs, pad_1, pad_2)
            layer = tf.compat.v1.layers.conv2d(
                padded_inputs, oc, ks, strides=st, padding='VALID', use_bias=False,
                dilation_rate=rate,
                kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1.0)),
                kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
            )
        layer = tf.compat.v1.layers.batch_normalization(layer, training=training)
    return layer


def conv_bn_relu(inputs, oc, ks, st, scope, training, rate=1):
    layer = conv_bn(inputs, oc, ks, st, scope, training, rate=rate)
    layer = tf.nn.relu(layer)
    return layer



def bottleneck(inputs, oc, st, scope, training, rate=1):
    with tf.compat.v1.variable_scope(scope):
        ic = inputs.get_shape().as_list()[-1]
        if ic == oc:
            if st == 1:
                shortcut = inputs
            else:
                shortcut = \
                tf.nn.max_pool2d(inputs, [1, st, st, 1], [1, st, st, 1], 'SAME')
        else:
            shortcut = conv_bn(inputs, oc, 1, st, 'shortcut', training)

        residual = conv_bn_relu(inputs, oc//4, 1, 1, 'conv1', training)
        residual = conv_bn_relu(residual, oc//4, 3, st, 'conv2', training, rate)
        residual = conv_bn(residual, oc, 1, 1, 'conv3', training)
        output = tf.nn.relu(shortcut + residual)

    return output


def resnet50(inputs, scope, training):
    with tf.compat.v1.variable_scope(scope):
        layer = conv_bn_relu(inputs, 64, 7, 2, 'conv1', training)

        with tf.compat.v1.variable_scope('block1'):
            for unit in range(2):
                layer = bottleneck(layer, 256, 1, 'unit%d' % (unit+1), training)
            layer = bottleneck(layer, 256, 2, 'unit3', training)

        with tf.compat.v1.variable_scope('block2'):
            for unit in range(4):
                layer = bottleneck(layer, 512, 1, 'unit%d' % (unit+1), training, 2)

        with tf.compat.v1.variable_scope('block3'):
            for unit in range(6):
                layer = bottleneck(layer, 1024, 1, 'unit%d' % (unit+1), training, 4)

        layer = conv_bn_relu(layer, 256, 3, 1, 'squeeze', training)

    return layer


def net_2d(features, training, scope, n_out):
    with tf.compat.v1.variable_scope(scope):
        layer = conv_bn_relu(features, 256, 3, 1, 'project', training)
        with tf.compat.v1.variable_scope('prediction'):
            hmap = tf.compat.v1.layers.conv2d(
            layer, n_out, 1, strides=1, padding='SAME',
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.compat.v1.initializers.truncated_normal(stddev=0.01)
        )
    return hmap


def net_3d(features, training, scope, n_out, need_norm):
    with tf.compat.v1.variable_scope(scope):
        layer = conv_bn_relu(features, 256, 3, 1, 'project', training)
        with tf.compat.v1.variable_scope('prediction'):
            dmap_raw = tf.compat.v1.layers.conv2d(
            layer, n_out * 3, 1, strides=1, padding='SAME',
            activation=None,
            kernel_initializer=tf.compat.v1.initializers.truncated_normal(stddev=0.01)
        )
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


def detnet(img, n_stack, training):
    features = resnet50(img, 'resnet', training)
    pos_tile = get_pose_tile(tf.shape(input=img)[0])
    features = tf.concat([features, pos_tile], -1)

    hmaps = []
    dmaps = []
    lmaps = []
    for i in range(n_stack):
        hmap = net_2d(features, training, 'hmap_%d' % i, 21)
        features = tf.concat([features, hmap], axis=-1)
        hmaps.append(hmap)

        dmap = net_3d(features, training, 'dmap_%d' % i, 21, False)
        features = tf.concat([features, tf.reshape(dmap, [-1, 32, 32, 21 * 3])], -1)
        dmaps.append(dmap)

        lmap = net_3d(features, training, 'lmap_%d' % i, 21, False)
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




class ModelDet:
    """
    DetNet: estimating 3D keypoint positions from input color image.
    """

    def __init__(self, model_path):
        """
    Parameters
    ----------
    model_path : str
      Path to the trained model.
    """
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.compat.v1.variable_scope('prior_based_hand'):
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.compat.v1.Session(config=config)
                self.input_ph = tf.compat.v1.placeholder(tf.uint8, [128, 128, 3])
                self.feed_img = \
                    tf.cast(tf.expand_dims(self.input_ph, 0), tf.float32) / 255
                self.hmaps, self.dmaps, self.lmaps = \
                    detnet(self.feed_img, 1, False)

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


class ModelPipeline:
    """
  A wrapper that puts DetNet and IKNet together.
  """

    def __init__(self, left=True):
        self.det_model = ModelDet(DETECTION_MODEL_PATH)

    def process(self, frame):
        """
        Process a single frame.

        Parameters
        ----------
        frame : np.ndarray, shape [128, 128, 3], dtype np.uint8.
          Frame to be processed.

        Returns
        -------
        np.ndarray, shape [21, 3]
          Joint locations.
        np.ndarray, shape [21, 4]
          Joint rotations.
        """
        xyz, uv = self.det_model.process(frame)
        return xyz, uv
