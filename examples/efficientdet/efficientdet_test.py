import tensorflow as tf
import efficientnet_model
from efficientnet_builder import build_backbone
from efficientdet import EFFICIENTDETD0
from efficientdet_postprocess import merge_class_box_level_outputs


def get_test_images(image_size, batch_size=1):
    images = tf.zeros((batch_size,
                       image_size,
                       image_size,
                       3),
                      dtype=tf.float32)
    return images


def test_efficientdet_model():
    detector = EFFICIENTDETD0()
    image_size = detector.image_size
    max_level = detector.max_level
    min_level = detector.min_level
    num_levels = max_level - min_level + 1
    images = get_test_images(image_size)
    class_outs, det_outs = detector(images)
    assert len(class_outs) == num_levels, 'Class outputs length fail'
    assert len(det_outs) == num_levels, 'Box outputs length fail'


def test_efficientdet_merge_outputs():
    detector = EFFICIENTDETD0()
    image_size = detector.image_size
    max_level = detector.max_level
    min_level = detector.min_level
    num_levels = max_level - min_level + 1
    num_classes = detector.num_classes
    images = get_test_images(image_size)
    class_outs, box_outs = detector(images)
    class_outs_expected = tf.zeros((1, 49104, num_classes), dtype=tf.float32)
    box_outs_expected = tf.zeros((1, 49104, 4), dtype=tf.float32)
    class_outs, box_outs = merge_class_box_level_outputs(class_outs,
                                                         box_outs,
                                                         num_levels,
                                                         num_classes)
    assert class_outs_expected.shape == class_outs.shape, \
        'Merged class outputs not matching'
    assert box_outs_expected.shape == box_outs.shape,\
        'Merged box outputs not matching'


def test_efficientnet_model():
    backbone = build_backbone('efficientnet-b0', 'swish', None)
    detector = EFFICIENTDETD0()
    image_size = detector.image_size
    images = get_test_images(image_size)
    features = backbone(images, False, True)
    assert len(features) == 6, 'EfficientNet model features length mismatch'


def test_efficientnet_se_block():
    images = get_test_images(128, 10)
    global_params = efficientnet_model.GlobalParams(
        batch_norm=tf.keras.layers.BatchNormalization, num_classes=10,
        dropout_rate=0, activation='swish', use_se=True)
    block_args = [efficientnet_model.BlockArgs(
        kernel_size=3, num_repeat=3, input_filters=3, output_filters=6,
        expand_ratio=6, id_skip=False, strides=[2, 2], se_ratio=0.8,
        conv_type=0, fused_conv=0, super_pixel=0)]
    model = efficientnet_model.Model(block_args, global_params)
    outputs = model(images, True)
    assert ((10, 10)) == outputs[0].shape, \
        'EfficientNet SE Block output shape mismatch'


test_efficientdet_model()
test_efficientdet_merge_outputs()
test_efficientnet_model()
test_efficientnet_se_block()
