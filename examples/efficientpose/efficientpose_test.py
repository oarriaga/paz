import pytest
import numpy as np
import tensorflow as tf
from losses import MultiPoseLoss
from efficientpose import EFFICIENTPOSEA


@pytest.fixture
def dataset_path():
    return 'Linemod_preprocessed/'


def test_multi_pose_loss_zero_condition(dataset_path):
    model = EFFICIENTPOSEA()
    pose_loss = MultiPoseLoss('08', model.translation_priors, dataset_path)
    num_anchors = model.translation_priors.shape[0]
    target_shape = (1, num_anchors, 11)
    prediction_shape = (1, num_anchors, 6)
    y_true = np.zeros(target_shape, dtype=np.float32)
    y_pred = np.zeros(prediction_shape, dtype=np.float32)
    loss = pose_loss.compute_loss(y_true, y_pred)
    assert tf.is_tensor(loss), 'Incorrect loss datatype'
    assert loss.shape == [], 'Incorrect loss shape'
    assert not tf.math.reduce_any(tf.math.is_nan(loss)), 'Loss has NaN values'
    assert loss.numpy() == 0.0, 'Incorrect loss value'
    del model


@pytest.mark.parametrize(('num_anchors, loss'),
                         [
                            (10,   1207.7874),
                            (100,  1178.727),
                            (1000, 1111.5929),
                            (500,  1111.9001),
                            (5000, 1094.8196),
                         ])
def test_multi_pose_loss_non_zero_condition(dataset_path, num_anchors, loss):
    model = EFFICIENTPOSEA()
    pose_loss = MultiPoseLoss('08', model.translation_priors[:num_anchors, :],
                              dataset_path)
    prediction_shape = (1, num_anchors, 6)
    rotation = tf.ones((num_anchors, 3), dtype=tf.float32)
    is_symmetric = tf.zeros((num_anchors, 1), dtype=tf.float32)
    flag = tf.ones((num_anchors, 1), dtype=tf.float32)
    y_true = tf.concat([rotation, is_symmetric, is_symmetric,
                        flag, rotation, flag, flag], axis=1)
    y_true = tf.expand_dims(y_true, axis=0)
    y_pred = tf.ones(prediction_shape, dtype=np.float32)
    loss = pose_loss.compute_loss(y_true, y_pred)

    assert tf.is_tensor(loss), 'Incorrect loss datatype'
    assert loss.shape == [], 'Incorrect loss shape'
    assert not tf.math.reduce_any(tf.math.is_nan(loss)), 'Loss has NaN values'
    assert tf.experimental.numpy.allclose(loss.numpy(), loss), (
        'Incorrect loss value')
    del model


@pytest.mark.parametrize(('num_anchors'),
                         [
                            (10),
                            (100),
                            (1000),
                            (500),
                            (5000),
                         ])
def test_loss_gradients(dataset_path, num_anchors):
    model = EFFICIENTPOSEA()
    pose_loss = MultiPoseLoss('08', model.translation_priors[:num_anchors, :],
                              dataset_path)
    prediction_shape = (1, num_anchors, 6)
    rotation = tf.ones((num_anchors, 3), dtype=tf.float32)
    is_symmetric = tf.zeros((num_anchors, 1), dtype=tf.float32)
    flag = tf.ones((num_anchors, 1), dtype=tf.float32)
    y_true = tf.concat([rotation, is_symmetric, is_symmetric,
                        flag, rotation, flag, flag], axis=1)
    y_true = tf.expand_dims(y_true, axis=0)
    y_pred = tf.ones(prediction_shape, dtype=np.float32)

    with tf.GradientTape(persistent=True) as g:
        g.watch(y_pred)
        loss = pose_loss.compute_loss(y_true, y_pred)
    gradients = g.gradient(loss, y_pred)
    assert not tf.math.reduce_any(tf.math.is_nan(gradients)), (
        'Gradients have NaN values')
    del model
