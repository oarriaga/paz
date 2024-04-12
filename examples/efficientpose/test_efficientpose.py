import pytest
import numpy as np
import tensorflow as tf
from paz.models.pose_estimation import (EfficientPosePhi0, EfficientPosePhi1,
                                        EfficientPosePhi2, EfficientPosePhi3,
                                        EfficientPosePhi4, EfficientPosePhi5,
                                        EfficientPosePhi6, EfficientPosePhi7)
from paz.datasets.linemod import LINEMOD_CAMERA_MATRIX as camera_matrix
from processors import RegressTranslation, ComputeTxTyTz
from anchors import build_translation_anchors
from losses import MultiPoseLoss


@pytest.fixture()
def get_camera_matrix():
    return camera_matrix


@pytest.fixture
def dataset_path():
    return 'Linemod_preprocessed/'


@pytest.fixture
def model_input_name():
    return 'image'


@pytest.fixture
def model_output_name():
    return ['boxes', 'transformation']


def get_test_images(image_size, batch_size=1):
    """Generates a simple mock image.

    # Arguments
        image_size: Int, integer value for H x W image shape.
        batch_size: Int, batch size for the input tensor.

    # Returns
        image: Zeros of shape (batch_size, H, W, C)
    """
    return tf.zeros((batch_size, image_size, image_size, 3), dtype=tf.float32)


@pytest.mark.parametrize(
        ('model'), [EfficientPosePhi0, EfficientPosePhi1, EfficientPosePhi2,
                    EfficientPosePhi3, EfficientPosePhi4, EfficientPosePhi5,
                    EfficientPosePhi6, EfficientPosePhi7])
def test_RegressTranslation(model):
    model = model(build_translation_anchors, num_classes=2,
                  base_weights='COCO', head_weights=None)
    regress_translation = RegressTranslation(model.translation_priors)
    translation_raw = np.zeros_like(model.translation_priors)
    translation_raw = np.expand_dims(translation_raw, axis=0)
    translation = regress_translation(translation_raw)
    assert translation[:, 0].sum() == model.translation_priors[:, 0].sum()
    assert translation[:, 1].sum() == model.translation_priors[:, 1].sum()
    assert translation[:, 2].sum() == translation_raw[:, :, 2].sum()
    del model


@pytest.mark.parametrize(
        ('model'), [EfficientPosePhi0, EfficientPosePhi1, EfficientPosePhi2,
                    EfficientPosePhi3, EfficientPosePhi4, EfficientPosePhi5,
                    EfficientPosePhi6, EfficientPosePhi7])
def test_ComputeTxTyTz(model, get_camera_matrix):
    model = model(build_translation_anchors, num_classes=2,
                  base_weights='COCO', head_weights=None)
    translation_raw = np.zeros_like(model.translation_priors)
    compute_tx_ty_tz = ComputeTxTyTz()
    tx_ty_tz = compute_tx_ty_tz(translation_raw, get_camera_matrix, 0.8)
    assert tx_ty_tz.shape == model.translation_priors.shape
    del model


def test_multi_pose_loss_zero_condition(dataset_path):
    model = EfficientPosePhi0(build_translation_anchors, num_classes=2,
                              base_weights='COCO', head_weights=None)
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
    model = EfficientPosePhi0(build_translation_anchors, num_classes=2,
                              base_weights='COCO', head_weights=None)
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
    model = EfficientPosePhi0(build_translation_anchors, num_classes=2,
                              base_weights='COCO', head_weights=None)
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


@pytest.mark.parametrize(('model, model_name, trainable_parameters,'
                          'non_trainable_parameters, input_shape,'
                          'output_shape'),
                         [
                            (EfficientPosePhi0, 'EfficientPose-Phi0', 4048025,
                                47136, (512, 512, 3), 49104),
                            (EfficientPosePhi1, 'EfficientPose-Phi1', 4279724,
                                51424, (640, 640, 3), 76725),
                            (EfficientPosePhi2, 'EfficientPose-Phi2', 8248005,
                                81776, (768, 768, 3), 110484),
                            (EfficientPosePhi3, 'EfficientPose-Phi3', 12436426,
                                114304, (896, 896, 3), 150381),
                            (EfficientPosePhi4, 'EfficientPose-Phi4', 14410717,
                                129920, (1024, 1024, 3), 196416),
                            (EfficientPosePhi5, 'EfficientPose-Phi5', 34012133,
                                227392, (1280, 1280, 3), 306900),
                            (EfficientPosePhi6, 'EfficientPose-Phi6', 52564204,
                                311984, (1280, 1280, 3), 306900),
                            (EfficientPosePhi7, 'EfficientPose-Phi7', 52564204,
                                311984, (1536, 1536, 3), 441936),
                         ])
def test_EfficientPose_architecture(model, model_name, model_input_name,
                                    model_output_name, trainable_parameters,
                                    non_trainable_parameters, input_shape,
                                    output_shape):
    implemented_model = model(build_translation_anchors, num_classes=2,
                              base_weights='COCO', head_weights=None)
    trainable_count = count_params(
        implemented_model.trainable_weights)
    non_trainable_count = count_params(
        implemented_model.non_trainable_weights)
    model_output_shape = [(None, output_shape, 6), (None, output_shape, 6)]
    assert implemented_model.name == model_name, "Model name incorrect"
    assert implemented_model.input_names[0] == model_input_name, (
        "Input name incorrect")
    assert implemented_model.output_names == model_output_name, (
        "Output name incorrect")
    assert trainable_count == trainable_parameters, (
        "Incorrect trainable parameters count")
    assert non_trainable_count == non_trainable_parameters, (
        "Incorrect non-trainable parameters count")
    assert implemented_model.input_shape[1:] == input_shape, (
        "Incorrect input shape")
    assert implemented_model.output_shape == model_output_shape, (
        "Incorrect output shape")
    del implemented_model


@pytest.mark.parametrize(('model'),
                         [
                            EfficientPosePhi0,
                            EfficientPosePhi1,
                            EfficientPosePhi2,
                            EfficientPosePhi3,
                            EfficientPosePhi4,
                            EfficientPosePhi5,
                            EfficientPosePhi6,
                            EfficientPosePhi7,
                         ])
def test_load_weights(model):
    detector = model(build_translation_anchors, num_classes=2,
                     base_weights='COCO', head_weights=None)
    del detector


@pytest.mark.parametrize(('model, input_shape, num_boxes'),
                         [
                            (EfficientPosePhi0, 512, 49104),
                            (EfficientPosePhi1, 640, 76725),
                            (EfficientPosePhi2, 768, 110484),
                            (EfficientPosePhi3, 896, 150381),
                            (EfficientPosePhi4, 1024, 196416),
                            (EfficientPosePhi5, 1280, 306900),
                            (EfficientPosePhi6, 1280, 306900),
                            (EfficientPosePhi7, 1536, 441936),
                         ])
def test_translation_anchors(model, input_shape, num_boxes):
    model = model(build_translation_anchors, num_classes=2,
                  base_weights='COCO', head_weights=None)
    anchors = model.translation_priors
    anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]
    assert np.logical_and(anchor_x >= 0, anchor_x <= input_shape).all(), (
        "Invalid x-coordinates of anchor centre")
    assert np.logical_and(anchor_y >= 0, anchor_y <= input_shape).all(), (
        "Invalid y-coordinates of anchor centre")
    assert np.round(np.mean(anchor_x), 2) == input_shape / 2.0, (
        "Anchor boxes asymmetrically distributed along X-direction")
    assert np.round(np.mean(anchor_y), 2) == input_shape / 2.0, (
        "Anchor boxes asymmetrically distributed along Y-direction")
    assert anchors.shape[0] == num_boxes, (
        "Incorrect number of anchor boxes")
    del model


def count_params(weights):
    """Count the total number of scalars composing the weights.
    This function is taken from the repository of [Keras]
    (https://github.com/keras-team/keras/blob/428ed9f03a0a0b2edc22d4ce29
     001857f617227c/keras/utils/layer_utils.py#L107)
    This is a patch and it should be removed eventually.

    # Arguments:
        weights: List, containing the weights
            on which to compute params.

    # Returns:
        Int, the total number of scalars composing the weights.
    """
    unique_weights = {id(w): w for w in weights}.values()
    unique_weights = [w for w in unique_weights if hasattr(w, "shape")]
    weight_shapes = [w.shape.as_list() for w in unique_weights]
    standardized_weight_shapes = [
        [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
    ]
    return int(sum(np.prod(p) for p in standardized_weight_shapes))
