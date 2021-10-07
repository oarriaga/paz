import tensorflow as tf
import efficientnet_model
from efficientnet_builder import build_backbone
from efficientdet import EFFICIENTDETD0


def get_test_images(image_size, batch_size=1):
    """
    Generates a simple mock image.

    # Arguments
        image_size: Int, integer value for H x W image shape.
        batch_size: Int, batch size for the input tensor.

    # Returns
        image: Zeros of shape (batch_size, H, W, C)
    """
    return tf.zeros((batch_size, image_size, image_size, 3),
                    dtype=tf.float32)


def test_efficientdet_model():
    detector = EFFICIENTDETD0()
    image_size = detector.image_size
    expected_output_shape = list(detector.prior_boxes.shape)
    expected_output_shape[1] = expected_output_shape[1] + detector.num_classes
    expected_output_shape = [1, ] + expected_output_shape
    images = get_test_images(image_size)
    output_shape = list(detector(images).shape)
    assert output_shape == expected_output_shape, 'Class outputs length fail'


def test_efficientnet_model():
    backbone = build_backbone('efficientnet-b0', 'swish', None)
    detector = EFFICIENTDETD0()
    image_size = detector.image_size
    images = get_test_images(image_size)
    features = backbone(images, False)
    assert len(features) == 6, 'EfficientNet model features length mismatch'


def test_efficientnet_bottleneck_block():
    detector = EFFICIENTDETD0()
    image_size = detector.image_size
    images = get_test_images(128, 10)
    backbone = efficientnet_model.EfficientNet(
        0, 1., 1., 1., detector.backbone_name, strides=[[2,2]],
        kernel_sizes=[3], num_repeats=[3], num_classes=10, input_filters=[3],
        output_filters=[6], expand_ratios=[6])
    output_shape = backbone(images, False)[0].shape
    expected_shape = ((10, 32, 32, 8))
    assert output_shape == expected_shape, 'SE Block output shape mismatch'


def test_efficientnet_se_block():
    detector = EFFICIENTDETD0()
    image_size = detector.image_size
    images = get_test_images(128, 10)
    backbone = efficientnet_model.EfficientNet(
        0, 1., 1., 1., detector.backbone_name, strides=[[2,2]],
        kernel_sizes=[3], num_repeats=[3], num_classes=10, input_filters=[3],
        output_filters=[6], expand_ratios=[6], squeeze_excite_ratio=0.8)
    output_shape = backbone(images, False)[0].shape
    expected_shape = ((10, 32, 32, 8))
    assert output_shape == expected_shape, 'SE Block output shape mismatch'

from examples.efficientdet.efficientdet_blocks import FeatureNode

def test_feature_fusion_sum():
    nodes1 = tf.constant([1, 3])
    nodes2 = tf.constant([1, 3])
    feature_node = FeatureNode(6, [3, 4], 3, True, True, True,
                               False, 'sum',None)
    output_node = feature_node.fuse_features([nodes1, nodes2])
    expected_node = tf.constant([2, 6], dtype=tf.int32)
    check_equality = tf.math.equal(output_node, expected_node)
    check_flag = tf.reduce_all(check_equality)
    assert check_flag, 'Feature fusion - \'sum\' mismatch'


def test_feature_fusion_attention():
    nodes1 = tf.constant([1, 3], dtype=tf.float32)
    nodes2 = tf.constant([1, 3], dtype=tf.float32)
    feature_node = FeatureNode(6, [3, 4], 3, True, True, True,
                               False, 'attention',None)
    feature_node.build((10, 128, 128, 3))
    output_node = feature_node.fuse_features([nodes1, nodes2])
    expected_node = tf.constant([1.0, 3.0], dtype=tf.float32)
    check_equality = tf.math.equal(output_node, expected_node)
    check_flag = tf.reduce_all(check_equality)
    assert check_flag, 'Feature fusion - attention method mismatch'


def test_feature_fusion_fastattention():
    nodes1 = tf.constant([1, 3], dtype=tf.float32)
    nodes2 = tf.constant([1, 3], dtype=tf.float32)
    feature_node = FeatureNode(6, [3, 4], 3, True, True, True,
                               False, 'fastattention',None)
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