from statistics import mode
import tensorflow as tf
from tensorflow.keras import backend as K
from efficientnet_model import EfficientNet
from efficientdet import (EFFICIENTDETD0, EFFICIENTDETD1, EFFICIENTDETD2,
                          EFFICIENTDETD3, EFFICIENTDETD4, EFFICIENTDETD5,
                          EFFICIENTDETD6, EFFICIENTDETD7)
# from examples.efficientdet.efficientdet_blocks import FeatureNode
import pytest
from tensorflow.keras.models import model_from_json
from efficientnet_model import conv_normal_initializer
from efficientdet_blocks import FuseFeature


@pytest.fixture
def get_models_base_path():
    return ("/home/manummk95/Desktop/efficientdet_working/"
            "required/efficientdet_architectures/")


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


def test_efficientnet_model():
    image_size = 512
    images = get_test_images(image_size)
    features = EfficientNet(images, 'efficientnet-b0', (512, 512, 3))
    assert len(features) == 5, 'EfficientNet model features length mismatch'


def test_efficientnet_bottleneck_block():
    images = get_test_images(128, 10)
    output_shape = EfficientNet(
        images, 'efficientnet-b0', (128, 10), strides=[[2, 2]],
        kernel_sizes=[3], num_repeats=[3], intro_filters=[3],
        outro_filters=[6], expand_ratios=[6])[0].shape
    expected_shape = (10, 32, 32, 8)
    assert output_shape == expected_shape, 'SE Block output shape mismatch'


def test_efficientnet_se_block():
    images = get_test_images(128, 10)
    output_shape = EfficientNet(
        images, 'efficientnet-b0', (128, 10), strides=[[2, 2]],
        kernel_sizes=[3], num_repeats=[3], intro_filters=[3],
        outro_filters=[6], expand_ratios=[6],
        squeeze_excite_ratio=0.8)[0].shape
    expected_shape = (10, 32, 32, 8)
    assert output_shape == expected_shape, 'SE Block output shape mismatch'


def test_all_efficientdet_models(get_models_base_path):

    implemented_models = [EFFICIENTDETD0, EFFICIENTDETD1, EFFICIENTDETD2,
                          EFFICIENTDETD3, EFFICIENTDETD4, EFFICIENTDETD5,
                          EFFICIENTDETD6, EFFICIENTDETD7]

    custom_objects = {"conv_normal_initializer": conv_normal_initializer,
                      "FuseFeature": FuseFeature}
    for model_id, implemented_model in enumerate(implemented_models):
        K.clear_session()
        reference_model_path = (get_models_base_path + 'EFFICIENTDETD' +
                                str(model_id) + '.json')
        reference_model_file = open(reference_model_path, 'r')
        loaded_model_json = reference_model_file.read()
        reference_model_file.close()
        reference_model = model_from_json(loaded_model_json,
                                          custom_objects=custom_objects)
        K.clear_session()
        assert implemented_model().get_config() == reference_model.get_config()


# def test_feature_fusion_sum():
#     nodes1 = tf.constant([1, 3])
#     nodes2 = tf.constant([1, 3])
#     feature_node = FeatureNode(6, [3, 4], 3, True, True, True, False, 'sum',
#                                None)
#     output_node = feature_node.fuse_features([nodes1, nodes2])
#     expected_node = tf.constant([2, 6], dtype=tf.int32)
#     check_equality = tf.math.equal(output_node, expected_node)
#     check_flag = tf.reduce_all(check_equality)
#     assert check_flag, 'Feature fusion - \'sum\' mismatch'


# def test_feature_fusion_attention():
#     nodes1 = tf.constant([1, 3], dtype=tf.float32)
#     nodes2 = tf.constant([1, 3], dtype=tf.float32)
#     feature_node = FeatureNode(6, [3, 4], 3, True, True, True, False,
#                                'attention', None)
#     feature_node.build((10, 128, 128, 3))
#     output_node = feature_node.fuse_features([nodes1, nodes2])
#     expected_node = tf.constant([1.0, 3.0], dtype=tf.float32)
#     check_equality = tf.math.equal(output_node, expected_node)
#     check_flag = tf.reduce_all(check_equality)
#     assert check_flag, 'Feature fusion - attention method mismatch'


# def test_feature_fusion_fastattention():
#     nodes1 = tf.constant([1, 3], dtype=tf.float32)
#     nodes2 = tf.constant([1, 3], dtype=tf.float32)
#     feature_node = FeatureNode(6, [3, 4], 3, True, True, True, False,
#                                'fastattention', None)
#     feature_node.build((10, 128, 128, 3))
#     output_node = feature_node.fuse_features([nodes1, nodes2])
#     expected_node = tf.constant([0.99995005, 2.9998503], dtype=tf.float32)
#     check_equality = tf.math.equal(output_node, expected_node)
#     check_flag = tf.reduce_all(check_equality)
#     assert check_flag, 'Feature fusion - fastattention method mismatch'


# test_efficientdet_model()
# test_efficientnet_model()
# test_efficientnet_bottleneck_block()
# test_efficientnet_se_block()
# test_all_efficientdet_models(get_model_path)
# test_feature_fusion_sum()
# test_feature_fusion_attention()
# test_feature_fusion_fastattention()
