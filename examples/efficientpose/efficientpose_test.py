import pytest
import numpy as np
import tensorflow as tf
from losses import MultiPoseLoss
from efficientpose import EfficientPosePhi0
from tensorflow.keras.layers import Input
from paz.models.detection.efficientdet.efficientnet import EFFICIENTNET
from paz.models.detection.efficientdet.efficientdet_blocks import (
    EfficientNet_to_BiFPN, BiFPN)
from efficientpose_blocks import RotationNet


@pytest.fixture
def dataset_path():
    return 'Linemod_preprocessed/'


def test_multi_pose_loss_zero_condition(dataset_path):
    model = EfficientPosePhi0(num_classes=2, base_weights='COCO',
                              head_weights=None)
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
    model = EfficientPosePhi0(num_classes=2, base_weights='COCO',
                              head_weights=None)
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
    model = EfficientPosePhi0(num_classes=2, base_weights='COCO',
                              head_weights=None)
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


@pytest.mark.parametrize(('input_shape, scaling_coefficients, FPN_num_filters,'
                          'FPN_cell_repeats, fusion, subnet_iterations,'
                          'subnet_repeats, output_shapes'),
                         [
                            (512,  (1.0, 1.0, 0.8), 64, 3, 'fast', 1, 2,
                                (36864, 9216, 2304, 576, 144)),
                            (640,  (1.0, 1.1, 0.8), 88, 4, 'fast', 1, 2,
                                (57600, 14400, 3600, 900, 225)),
                            (768,  (1.1, 1.2, 0.7), 112, 5, 'fast', 1, 2,
                                (82944, 20736, 5184, 1296, 324)),
                            (896,  (1.2, 1.4, 0.7), 160, 6, 'fast', 2, 3,
                                (112896, 28224, 7056, 1764, 441)),
                            (1024, (1.4, 1.8, 0.6), 224, 7, 'fast', 2, 3,
                                (147456, 36864, 9216, 2304, 576)),
                            (1280, (1.6, 2.2, 0.6), 288, 7, 'fast', 2, 3,
                                (230400, 57600, 14400, 3600, 900)),
                            (1280, (1.8, 2.6, 0.5), 384, 8, 'sum', 3, 4,
                                (230400, 57600, 14400, 3600, 900)),
                            (1536, (1.8, 2.6, 0.5), 384, 8, 'sum', 3, 4,
                                (331776, 82944, 20736, 5184, 1296))
                         ])
def test_EfficientPose_RotationNet(input_shape, scaling_coefficients,
                                   FPN_num_filters, FPN_cell_repeats, fusion,
                                   subnet_iterations, subnet_repeats,
                                   output_shapes):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    branches, middles, skips = EfficientNet_to_BiFPN(
        branch_tensors, FPN_num_filters)
    for _ in range(FPN_cell_repeats):
        middles, skips = BiFPN(middles, skips, FPN_num_filters, fusion)
    num_dims, num_anchors, num_filters = (3, 9, 64)
    args = (middles, subnet_iterations, subnet_repeats, num_anchors)
    rotations = RotationNet(*args, num_filters, num_dims)
    assert len(rotations) == 5, 'Rotation output length fail'
    for rotation, output_shape in zip(rotations, output_shapes):
        assert rotation.shape == (None, output_shape, 3), (
            'Rotation outputs shape fail')
    del branch_tensors, branches, middles, skips, rotations
