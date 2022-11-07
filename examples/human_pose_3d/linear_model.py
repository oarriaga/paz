"""Simple model to regress 3d human poses from 2d joint locations"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    ReLU,
    Dense,
    Dropout,
    Input
)


def two_linear(xin, linear_size, residual, dropout_keep_prob, max_norm, batch_norm, idx):
    """Make a bi-linear block with optional residual connection

    Args
        xin: the batch that enters the block
        linear_size: integer. The size of the linear units
        residual: boolean. Whether to add a residual connection
        dropout_keep_prob: float [0,1]. Probability of dropping something out
        max_norm: boolean. Whether to clip weights to 1-norm
        batch_norm: boolean. Whether to do batch normalization
        idx: integer. Number of layer (for naming/scoping)
    Returns
        y: the batch after it leaves the block
    """
    initializer = tf.keras.initializers.HeNormal()
    # Linear 1
    if max_norm:
        y = Dense(linear_size, use_bias=True, kernel_initializer=initializer, bias_initializer=initializer,
                  kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), name="linear2_" + str(idx))(xin)
    else:
        y = Dense(linear_size, use_bias=True, kernel_initializer=initializer, bias_initializer=initializer,
                  name="linear2_" + str(idx))(xin)

    if batch_norm:
        y = BatchNormalization(name="batch_normalization1" + str(idx))(y) # , training=isTraining

    y = ReLU()(y)
    y = Dropout(dropout_keep_prob)(y)

    # Linear 2
    if max_norm:
        y = Dense(linear_size, use_bias=True, kernel_initializer=initializer, bias_initializer=initializer,
                  kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), name="linear3_" + str(idx))(y)
    else:
        y = Dense(linear_size, use_bias=True, kernel_initializer=initializer, bias_initializer=initializer,
                  name="linear3_" + str(idx))(y)

    if batch_norm:
        y = BatchNormalization(name="batch_normalization2" + str(idx))(y)

    y = ReLU()(y)
    y = Dropout(dropout_keep_prob)(y)

    # Residual every 2 blocks
    y = (xin + y) if residual else y

    return y


def get_all_batches(data_x, data_y, camera_frame, batch_size):
    """Obtain a list of all the batches, randomly permuted

    Args
        data_x: dictionary with 2d inputs
        data_y: dictionary with 3d expected outputs
        camera_frame: whether the 3d data is in camera coordinates

    Returns
        encoder_inputs: list of 2d batches
        decoder_outputs: list of 3d batches
    """
    input_size = 16*2
    output_size = 16*3
    # Figure out how many frames we have
    n = 0
    for key2d in data_x.keys():
        n2d, _ = data_x[key2d].shape
        n = n + n2d

    encoder_inputs = np.zeros((n, input_size), dtype=float)
    decoder_outputs = np.zeros((n, output_size), dtype=float)

    # Put all the data into big arrays
    idx = 0
    for key2d in data_x.keys():
        (subj, b, fname) = key2d
        # keys should be the same if 3d is in camera coordinates
        key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
        key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and camera_frame else key3d

        n2d, _ = data_x[key2d].shape
        encoder_inputs[idx:idx + n2d, :] = data_x[key2d]
        decoder_outputs[idx:idx + n2d, :] = data_y[key3d]
        idx = idx + n2d

    # Make the number of examples a multiple of the batch size
    n_extra = n % batch_size
    if n_extra > 0:  # Otherwise examples are already a multiple of batch size
        encoder_inputs = encoder_inputs[:-n_extra, :]
        decoder_outputs = decoder_outputs[:-n_extra, :]

    return encoder_inputs, decoder_outputs


def mse_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return loss


def LinearModel(
                linear_size,
                num_layers,
                residual,
                max_norm,
                batch_norm,
                dropout_keep_prob,
                predict_14,
                input_shape=(32,)
                 ):
    HUMAN_2D_SIZE = 16 * 2
    HUMAN_3D_SIZE = 14 * 3 if predict_14 else 16 * 3

    inputs = Input(shape=input_shape)
    initializer = tf.keras.initializers.HeNormal()
    if max_norm:
        y3 = Dense(linear_size, use_bias=True, kernel_initializer=initializer, bias_initializer=initializer,
                   kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), name="linear1_")(inputs)
    else:
        y3 = Dense(linear_size, use_bias=True, kernel_initializer=initializer, bias_initializer=initializer,
                   name="linear1_")(inputs)
    if batch_norm:
        y3 = BatchNormalization(name="batch_normalization")(y3)
    y3 = ReLU()(y3)
    y3 = Dropout(dropout_keep_prob)(y3)

    # === Create multiple bi-linear layers ===
    for idx in range(num_layers):
        y3 = two_linear(y3, linear_size, residual, dropout_keep_prob, max_norm, batch_norm, idx)

    # === Last linear layer has HUMAN_3D_SIZE in output ===
    if max_norm:
        y = Dense(HUMAN_3D_SIZE, use_bias=True, kernel_initializer=initializer, bias_initializer=initializer,
                  kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), name="linear4_")(y3)
    else:
        y = Dense(HUMAN_3D_SIZE, use_bias=True, kernel_initializer=initializer, bias_initializer=initializer,
                  name="linear4_")(y3)
    # === End linear model ===
    model = Model(inputs, outputs=y, name='LinearModel')
    return model

