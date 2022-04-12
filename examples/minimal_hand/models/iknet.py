
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import truncated_normal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def dense(layer, n_units):
    layer = tf.keras.layers.Dense(
        n_units, activation=None, kernel_regularizer=l2(0.5 * (1.0)),
        kernel_initializer=truncated_normal(stddev=0.01))(layer)
    return layer


def block(layer, n_units, training):
    layer = dense(layer, n_units)
    layer = BatchNormalization()(layer, training)
    return layer


def ModelIK(input_shape=(84, 3), num_keypoints=21, depth=6, width=1024):
    input = Input(shape=input_shape, dtype=tf.float32)
    layer = tf.reshape(input, [1, input_shape[0]*3])

    for arg in range(depth):
        layer = block(layer, width, training=False)
        layer = tf.sigmoid(layer)
    theta_raw = dense(layer, num_keypoints * 4)
    theta_raw = tf.reshape(theta_raw, [-1, num_keypoints, 4])
    norm = tf.norm(tensor=theta_raw, axis=-1, keepdims=True)
    eps = np.finfo(np.float32).eps
    norm = tf.maximum(norm, eps)

    theta_positive = theta_raw / norm
    theta_negative = theta_positive * -1
    theta = tf.where(tf.tile(theta_positive[:, :, 0:1] > 0, [1, 1, 4]),
                     theta_positive, theta_negative)

    model = Model(input, outputs=[theta])
    model_path = 'model_weights/iknet_weight.hdf5'
    model.load_weights(model_path)
    return model
