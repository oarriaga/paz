import pytest
import numpy as np
import tensorflow as tf
from keras.utils.layer_utils import count_params
from tensorflow.keras.layers import Input
from anchors import build_prior_boxes
from efficientdet import (EFFICIENTDETD0, EFFICIENTDETD1, EFFICIENTDETD2,
                          EFFICIENTDETD3, EFFICIENTDETD4, EFFICIENTDETD5,
                          EFFICIENTDETD6, EFFICIENTDETD7)
from efficientnet import EFFICIENTNET, apply_drop_connect


@pytest.fixture
def model_input_name():
    return 'image'


@pytest.fixture
def model_output_name():
    return 'boxes'


def get_test_images(image_size, batch_size=1):
    """Generates a simple mock image.

    # Arguments
        image_size: Int, integer value for H x W image shape.
        batch_size: Int, batch size for the input tensor.

    # Returns
        image: Zeros of shape (batch_size, H, W, C)
    """
    return tf.zeros((batch_size, image_size, image_size, 3), dtype=tf.float32)


@pytest.mark.parametrize(('input_shape, dtype, target_shape, is_training'),
                         [
                            ((1, 1), tf.dtypes.float64, ([1, 1, 1, 1]), True),
                            ((5, 5), tf.dtypes.float64, ([5, 1, 5, 5]), True),
                            ((3, 5), tf.dtypes.float64, ([3, 1, 3, 5]), True),
                            ((5, 3), tf.dtypes.float64, ([5, 1, 5, 3]), True),
                            ((1, 1), tf.dtypes.float64, (1, 1), False),
                            ((5, 5), tf.dtypes.float64, (5, 5), False),
                            ((3, 5), tf.dtypes.float64, (3, 5), False),
                            ((5, 3), tf.dtypes.float64, (5, 3), False)
                         ])
def test_drop_connect(input_shape, dtype, target_shape, is_training):
    x = tf.random.uniform(input_shape, minval=0, maxval=5, dtype=dtype)
    survival_rate = np.random.uniform(0.0, 1.0)
    y = apply_drop_connect(x, is_training, survival_rate)
    assert y.shape == target_shape, 'Incorrect target shape'
    assert y.dtype == dtype, 'Incorrect target datatype'


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


def test_EfficientNet_model():
    image_size = 512
    images = get_test_images(image_size)
    features = EFFICIENTNET(images, (1.0, 1.0, 0.8))
    assert len(features) == 5, 'EfficientNet model features length mismatch'
    del features


def test_EfficientNet_bottleneck_block():
    images = get_test_images(128, 10)
    output_shape = EFFICIENTNET(
        images, (1.0, 1.0, 0.8), strides=[[2, 2]],
        kernel_sizes=[3], repeats=[3], intro_filters=[3],
        outro_filters=[6], expand_ratios=[6])[0].shape
    expected_shape = (10, 32, 32, 8)
    assert output_shape == expected_shape, 'SE Block output shape mismatch'


def test_EfficientNet_se_block():
    images = get_test_images(128, 10)
    output_shape = EFFICIENTNET(
        images, (1.0, 1.0, 0.8), strides=[[2, 2]],
        kernel_sizes=[3], repeats=[3], intro_filters=[3],
        outro_filters=[6], expand_ratios=[6], SE_ratio=0.8)[0].shape
    expected_shape = (10, 32, 32, 8)
    assert output_shape == expected_shape, 'SE Block output shape mismatch'


@pytest.mark.parametrize(('input_shape, scaling_coefficients, feature_shape,'
                          'feature_channels'),
                         [
                            (512,  (1.0, 1.0, 0.8), (256, 128, 64, 32, 16),
                             (16, 24, 40, 112, 320)),
                            (640,  (1.0, 1.1, 0.8), (320, 160, 80, 40, 20),
                             (16, 24, 40, 112, 320)),
                            (768,  (1.1, 1.2, 0.7), (384, 192, 96, 48, 24),
                             (16, 24, 48, 120, 352)),
                            (896,  (1.2, 1.4, 0.7), (448, 224, 112, 56, 28),
                             (24, 32, 48, 136, 384)),
                            (1024, (1.4, 1.8, 0.6), (512, 256, 128, 64, 32),
                             (24, 32, 56, 160, 448)),
                            (1280, (1.6, 2.2, 0.6), (640, 320, 160, 80, 40),
                             (24, 40, 64, 176, 512)),
                            (1280, (1.8, 2.6, 0.5), (640, 320, 160, 80, 40),
                             (32, 40, 72, 200, 576)),
                            (1536, (1.8, 2.6, 0.5), (768, 384, 192, 96, 48),
                             (32, 40, 72, 200, 576))
                         ])
def test_EfficientNet_features(input_shape, scaling_coefficients,
                               feature_shape, feature_channels):
    shape = (input_shape, input_shape, 3)
    image = Input(shape=shape, name='image')
    branch_tensors = EFFICIENTNET(image, scaling_coefficients)
    assert len(branch_tensors) == 5, "Number of features mismatch"
    for branch_tensor, feature_shape_per_tensor, feature_channel  \
            in zip(branch_tensors, feature_shape, feature_channels):
        target_shape = (None, feature_shape_per_tensor,
                        feature_shape_per_tensor, feature_channel)
        assert branch_tensor.shape == target_shape, ("Shape of features"
                                                     "mismatch")
    del branch_tensors


@pytest.mark.parametrize(('model, model_name, trainable_parameters,'
                          'non_trainable_parameters, input_shape,'
                          'output_shape'),
                         [
                            (EFFICIENTDETD0, 'efficientdet-d0', 3880067,
                                47136, (512, 512, 3), (49104, 94)),
                            (EFFICIENTDETD1, 'efficientdet-d1', 6625898,
                                71456, (640, 640, 3), (76725, 94)),
                            (EFFICIENTDETD2, 'efficientdet-d2', 8097039,
                                81776, (768, 768, 3), (110484, 94)),
                            (EFFICIENTDETD3, 'efficientdet-d3', 12032296,
                                114304, (896, 896, 3), (150381, 94)),
                            (EFFICIENTDETD4, 'efficientdet-d4', 20723675,
                                167312, (1024, 1024, 3), (196416, 94)),
                            (EFFICIENTDETD5, 'efficientdet-d5', 33653315,
                                227392, (1280, 1280, 3), (306900, 94)),
                            (EFFICIENTDETD6, 'efficientdet-d6', 51871934,
                                311984, (1280, 1280, 3), (306900, 94)),
                            (EFFICIENTDETD7, 'efficientdet-d7', 51871934,
                                311984, (1536, 1536, 3), (441936, 94)),
                         ])
def test_efficientdet_architecture(model, model_name, model_input_name,
                                   model_output_name, trainable_parameters,
                                   non_trainable_parameters, input_shape,
                                   output_shape):
    implemented_model = model()
    trainable_count = count_params(
        implemented_model.trainable_weights)
    non_trainable_count = count_params(
        implemented_model.non_trainable_weights)
    assert implemented_model.name == model_name, "Model name incorrect"
    assert implemented_model.input_names[0] == model_input_name, (
        "Input name incorrect")
    assert implemented_model.output_names[0] == model_output_name, (
        "Output name incorrect")
    assert trainable_count == trainable_parameters, (
        "Incorrect trainable parameters count")
    assert non_trainable_count == non_trainable_parameters, (
        "Incorrect non-trainable parameters count")
    assert implemented_model.input_shape[1:] == input_shape, (
        "Incorrect input shape")
    assert implemented_model.output_shape[1:] == output_shape, (
        "Incorrect output shape")
    del implemented_model


@pytest.mark.parametrize(('min_level, max_level, num_scales, aspect_ratios,'
                          'anchor_scale, image_size, anchor_count'),
                         [
                            (3, 7, 3, [1.0, 2.0, 0.5], 4.0, (512, 512),
                                49104),
                            (3, 7, 3, [1.0, 2.0, 0.5], 4.0, (640, 640),
                                76725),
                            (3, 7, 3, [1.0, 2.0, 0.5], 4.0, (768, 768),
                                110484),
                            (3, 7, 3, [1.0, 2.0, 0.5], 4.0, (896, 896),
                                150381),
                            (3, 7, 3, [1.0, 2.0, 0.5], 4.0, (1024, 1024),
                                196416),
                            (3, 7, 3, [1.0, 2.0, 0.5], 4.0, (1280, 1280),
                                306900),
                            (3, 7, 3, [1.0, 2.0, 0.5], 4.0, (1536, 1536),
                                441936),
                         ])
def test_build_prior_boxes(min_level, max_level, num_scales, aspect_ratios,
                           anchor_scale, image_size, anchor_count):
    prior_boxes = build_prior_boxes(
        min_level, max_level, num_scales, aspect_ratios, anchor_scale,
        image_size)
    anchor_x, anchor_y = prior_boxes[:, 0], prior_boxes[:, 1]
    anchor_W, anchor_H = prior_boxes[:, 2], prior_boxes[:, 3]
    measured_aspect_ratios = set(np.unique(np.round((anchor_W / anchor_H), 2)))
    assert np.logical_and(anchor_x >= 0, anchor_x <= 1).all(), (
        "Invalid x-coordinates of anchor centre")
    assert np.logical_and(anchor_y >= 0, anchor_y <= 1).all(), (
        "Invalid y-coordinates of anchor centre")
    assert (anchor_W > 0).all(), "Invalid/negative anchor width"
    assert (anchor_H > 0).all(), "Invalid/negative anchor height"
    assert np.round(np.mean(anchor_x), 2) == 0.5, (
        "Anchor boxes asymmetrically distributed along X-direction")
    assert np.round(np.mean(anchor_y), 2) == 0.5, (
        "Anchor boxes asymmetrically distributed along Y-direction")
    assert measured_aspect_ratios == set(aspect_ratios), (
        "Anchor aspect ratios not as expected")
    assert prior_boxes.shape[0] == anchor_count, (
        "Incorrect number of anchor boxes")
