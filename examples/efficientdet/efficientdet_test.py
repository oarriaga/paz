import tensorflow as tf
from efficientnet_model import EfficientNet
from efficientnet_builder import get_efficientnet_model
from efficientdet import EFFICIENTDETD0
from examples.efficientdet.efficientdet_blocks import FeatureNode


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
    backbone = get_efficientnet_model('efficientnet-b0')
    image_size = 512
    images = get_test_images(image_size)
    features = backbone(images, False)
    assert len(features) == 6, 'EfficientNet model features length mismatch'


def test_efficientnet_bottleneck_block():
    images = get_test_images(128, 10)
    backbone = EfficientNet(
        1., 1., 1., 'efficientnet-b0', strides=[[2, 2]], kernel_sizes=[3],
        num_repeats=[3], num_classes=10, input_filters=[3], output_filters=[6],
        expand_ratios=[6])
    output_shape = backbone(images, False)[0].shape
    expected_shape = (10, 32, 32, 8)
    assert output_shape == expected_shape, 'SE Block output shape mismatch'


def test_efficientnet_se_block():
    images = get_test_images(128, 10)
    backbone = EfficientNet(
        1., 1., 1., 'efficientnet-b0', strides=[[2, 2]], kernel_sizes=[3],
        num_repeats=[3], num_classes=10, input_filters=[3], output_filters=[6],
        expand_ratios=[6], squeeze_excite_ratio=0.8)
    output_shape = backbone(images, False)[0].shape
    expected_shape = (10, 32, 32, 8)
    assert output_shape == expected_shape, 'SE Block output shape mismatch'


def test_feature_fusion_sum():
    nodes1 = tf.constant([1, 3])
    nodes2 = tf.constant([1, 3])
    feature_node = FeatureNode(6, [3, 4], 3, True, True, True, False, 'sum',
                               None)
    output_node = feature_node.fuse_features([nodes1, nodes2])
    expected_node = tf.constant([2, 6], dtype=tf.int32)
    check_equality = tf.math.equal(output_node, expected_node)
    check_flag = tf.reduce_all(check_equality)
    assert check_flag, 'Feature fusion - \'sum\' mismatch'


def test_feature_fusion_attention():
    nodes1 = tf.constant([1, 3], dtype=tf.float32)
    nodes2 = tf.constant([1, 3], dtype=tf.float32)
    feature_node = FeatureNode(6, [3, 4], 3, True, True, True, False,
                               'attention', None)
    feature_node.build((10, 128, 128, 3))
    output_node = feature_node.fuse_features([nodes1, nodes2])
    expected_node = tf.constant([1.0, 3.0], dtype=tf.float32)
    check_equality = tf.math.equal(output_node, expected_node)
    check_flag = tf.reduce_all(check_equality)
    assert check_flag, 'Feature fusion - attention method mismatch'


def test_feature_fusion_fastattention():
    nodes1 = tf.constant([1, 3], dtype=tf.float32)
    nodes2 = tf.constant([1, 3], dtype=tf.float32)
    feature_node = FeatureNode(6, [3, 4], 3, True, True, True, False,
                               'fastattention', None)
    feature_node.build((10, 128, 128, 3))
    output_node = feature_node.fuse_features([nodes1, nodes2])
    expected_node = tf.constant([0.99995005, 2.9998503], dtype=tf.float32)
    check_equality = tf.math.equal(output_node, expected_node)
    check_flag = tf.reduce_all(check_equality)
    assert check_flag, 'Feature fusion - fastattention method mismatch'


test_efficientdet_model()
test_efficientnet_model()
test_efficientnet_bottleneck_block()
test_efficientnet_se_block()
test_feature_fusion_sum()
test_feature_fusion_attention()
test_feature_fusion_fastattention()
