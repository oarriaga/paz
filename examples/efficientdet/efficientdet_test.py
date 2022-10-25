import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input

from efficientdet import (EFFICIENTDETD0, EFFICIENTDETD1, EFFICIENTDETD2,
                          EFFICIENTDETD3, EFFICIENTDETD4, EFFICIENTDETD5,
                          EFFICIENTDETD6, EFFICIENTDETD7)
from efficientdet_blocks import FuseFeature
from efficientnet_model import EfficientNet, conv_normal_initializer


@pytest.fixture
def models_base_path():
    return ("/home/manummk95/Desktop/efficientdet_working/"
            "required/test_files/efficientdet_architectures/")


@pytest.fixture
def model_input_output_base_path():
    return ("/home/manummk95/Desktop/efficientdet_working/"
            "required/test_files/test_model_outputs/")


def get_test_images(image_size, batch_size=1):
    """Generates a simple mock image.

    # Arguments
        image_size: Int, integer value for H x W image shape.
        batch_size: Int, batch size for the input tensor.

    # Returns
        image: Zeros of shape (batch_size, H, W, C)
    """
    return tf.zeros((batch_size, image_size, image_size, 3), dtype=tf.float32)


def test_efficientdet_model():
    detector = EFFICIENTDETD0()
    image_size = 512
    num_classes = 90
    expected_output_shape = list(detector.prior_boxes.shape)
    expected_output_shape[1] = expected_output_shape[1] + num_classes
    expected_output_shape = [1, ] + expected_output_shape
    images = get_test_images(image_size)
    output_shape = list(detector(images).shape)
    assert output_shape == expected_output_shape, 'Class outputs length fail'
    del detector


def test_efficientnet_model():
    image_size = 512
    images = get_test_images(image_size)
    features = EfficientNet(images, 'efficientnet-b0', (512, 512, 3))
    assert len(features) == 5, 'EfficientNet model features length mismatch'
    del features


def test_efficientnet_bottleneck_block():
    images = get_test_images(128, 10)
    output_shape = EfficientNet(
        images, 'efficientnet-b0', (128, 10), strides=[[2, 2]],
        kernel_sizes=[3], repeats=[3], intro_filters=[3],
        outro_filters=[6], expand_ratios=[6])[0].shape
    expected_shape = (10, 32, 32, 8)
    assert output_shape == expected_shape, 'SE Block output shape mismatch'


def test_efficientnet_se_block():
    images = get_test_images(128, 10)
    output_shape = EfficientNet(
        images, 'efficientnet-b0', (128, 10), strides=[[2, 2]],
        kernel_sizes=[3], repeats=[3], intro_filters=[3],
        outro_filters=[6], expand_ratios=[6],
        squeeze_excite_ratio=0.8)[0].shape
    expected_shape = (10, 32, 32, 8)
    assert output_shape == expected_shape, 'SE Block output shape mismatch'


@pytest.mark.parametrize(('input_shape, backbone, feature_shape,'
                          'feature_channels'),
                         [
                             (512,  'efficientnet-b0', (256, 128, 64, 32, 16),
                              (16, 24, 40, 112, 320)),
                             (640,  'efficientnet-b1', (320, 160, 80, 40, 20),
                              (16, 24, 40, 112, 320)),
                             (768,  'efficientnet-b2', (384, 192, 96, 48, 24),
                              (16, 24, 48, 120, 352)),
                             (896,  'efficientnet-b3', (448, 224, 112, 56, 28),
                              (24, 32, 48, 136, 384)),
                             (1024, 'efficientnet-b4', (512, 256, 128, 64, 32),
                              (24, 32, 56, 160, 448)),
                             (1280, 'efficientnet-b5', (640, 320, 160, 80, 40),
                              (24, 40, 64, 176, 512)),
                             (1280, 'efficientnet-b6', (640, 320, 160, 80, 40),
                              (32, 40, 72, 200, 576)),
                             (1536, 'efficientnet-b6', (768, 384, 192, 96, 48),
                              (32, 40, 72, 200, 576))
                         ])
def test_efficientnet_features(input_shape, backbone, feature_shape,
                               feature_channels):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EfficientNet(image, backbone, shape)
    assert len(branch_tensors) == 5, "Number of features mismatch"
    for branch_tensor, feature_shape_per_tensor, feature_channel  \
            in zip(branch_tensors, feature_shape, feature_channels):
        target_shape = (None, feature_shape_per_tensor,
                        feature_shape_per_tensor, feature_channel)
        assert branch_tensor.shape == target_shape, ("Shape of features"
                                                     "mismatch")
    del branch_tensors


@pytest.mark.parametrize('implemented_model, model_id',
                         [
                             (EFFICIENTDETD0, 0),
                             (EFFICIENTDETD1, 1),
                             (EFFICIENTDETD2, 2),
                             (EFFICIENTDETD3, 3),
                             (EFFICIENTDETD4, 4),
                             (EFFICIENTDETD5, 5),
                             (EFFICIENTDETD6, 6),
                             (EFFICIENTDETD7, 7),
                         ])
def test_efficientdet_architecture(models_base_path,
                                   implemented_model,
                                   model_id):
    custom_objects = {"conv_normal_initializer": conv_normal_initializer,
                      "FuseFeature": FuseFeature}
    K.clear_session()
    reference_model_path = (models_base_path + 'EFFICIENTDET-D' +
                            str(model_id) + '.hdf5')
    reference_model = tf.keras.models.load_model(
        reference_model_path, custom_objects=custom_objects)
    K.clear_session()
    assert (implemented_model().get_config() ==
            reference_model.get_config()), ('EFFICIENTDETD' + str(model_id)
                                            + " architecture mismatch")
    del implemented_model, reference_model


@pytest.mark.parametrize('model',
                         [
                             (EFFICIENTDETD0),
                             (EFFICIENTDETD1),
                             (EFFICIENTDETD2),
                             (EFFICIENTDETD3),
                             (EFFICIENTDETD4),
                             (EFFICIENTDETD5),
                             (EFFICIENTDETD6),
                             (EFFICIENTDETD7),
                         ])
def test_efficientdet_anchor_boxes(model):
    model_anchor = model().prior_boxes
    anchor_x, anchor_y = model_anchor[:, 0], model_anchor[:, 1]
    anchor_W, anchor_H = model_anchor[:, 2], model_anchor[:, 3]
    assert np.logical_and(anchor_x >= 0, anchor_x <= 1).all()
    assert np.logical_and(anchor_y >= 0, anchor_y <= 1).all()
    assert (anchor_W > 0).all()
    assert (anchor_H > 0).all()
    del model


@pytest.mark.skip(reason="Training of model needs to carried out from D0-D7")
@pytest.mark.parametrize('model, model_idx, preprocessed_inputs',
                         [
                             (EFFICIENTDETD0, 0, (1, 2, 3, 4, 5)),
                             (EFFICIENTDETD1, 1, (1, 2, 3, 4, 5)),
                             (EFFICIENTDETD2, 2, (1, 2, 3, 4, 5)),
                             (EFFICIENTDETD3, 3, (1, 2, 3, 4, 5)),
                             (EFFICIENTDETD4, 4, (1, 2, 3, 4, 5)),
                             (EFFICIENTDETD5, 5, (1, 2, 3, 4, 5)),
                             (EFFICIENTDETD6, 6, (1, 2, 3, 4, 5)),
                             (EFFICIENTDETD7, 7, (1, 2, 3, 4, 5))
                         ])
def test_efficientdet_result(model_input_output_base_path, model,
                             model_idx, preprocessed_inputs):
    for preprocessed_input_idx in preprocessed_inputs:
        preprocessed_input_file = model_input_output_base_path + \
            'EFFICIENTDETD' + str(model_idx) + '/inputs/test_image_' + \
            str(preprocessed_input_idx) + '.npy'
        with open(preprocessed_input_file, 'rb') as f:
            preprocessed_input = np.load(f)

        target_model_output_file = model_input_output_base_path + \
            'EFFICIENTDETD' + str(model_idx) + '/outputs/model_output_' + \
            str(preprocessed_input_idx) + '.npy'
        with open(target_model_output_file, 'rb') as f:
            target_model_output = np.load(target_model_output_file)

        assert np.all(model()(preprocessed_input).numpy() ==
                      target_model_output), 'Model result not as expected'
    del model
