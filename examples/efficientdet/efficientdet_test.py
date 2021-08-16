import tensorflow as tf
import efficientnet_model
from efficientnet_builder import build_backbone
from efficientdet import EFFICIENTDETD0
from efficientdet_postprocess import merge_level_outputs


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
    detector = EFFICIENTDETD0(return_base=True)
    image_size = detector.image_size
    images = get_test_images(image_size)
    max_level = detector.max_level
    min_level = detector.min_level
    num_levels = max_level - min_level + 1
    class_outs, det_outs = detector(images)
    assert len(class_outs) == num_levels, 'Class outputs length fail'
    assert len(det_outs) == num_levels, 'Box outputs length fail'


def test_efficientdet_merge_outputs():
    detector = EFFICIENTDETD0(return_base=True)
    image_size = detector.image_size
    images = get_test_images(image_size)
    max_level = detector.max_level
    min_level = detector.min_level
    num_levels = max_level - min_level + 1
    num_classes = detector.num_classes
    class_outs, box_outs = detector(images)
    class_outs_expected = tf.zeros((1, 49104, num_classes), dtype=tf.float32)
    box_outs_expected = tf.zeros((1, 49104, 4), dtype=tf.float32)
    class_outs, box_outs = merge_level_outputs(
        class_outs, box_outs, num_levels, num_classes)
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
    global_params = efficientnet_model.GlobalParams
    global_params["batch_norm"] = tf.keras.layers.BatchNormalization
    global_params["num_classes"] = 10
    global_params["dropout_rate"] = 0
    global_params["activation"] = 'swish'
    global_params["use_se"] = True
    block_args = efficientnet_model.BlockArgs
    block_args["kernel_size"] = 3
    block_args["num_repeat"] = 3
    block_args["input_filters"] = 3
    block_args["output_filters"] = 6
    block_args["expand_ratio"] = 6
    block_args["id_skip"] = False
    block_args["strides"] = [2, 2]
    block_args["se_ratio"] = 0.8
    block_args["conv_type"] = 0
    block_args["fused_conv"] = 0
    block_args["super_pixel"] = 0
    block_args = [block_args]
    model = efficientnet_model.Model(block_args, global_params)
    outputs = model(images, True)
    assert ((10, 10)) == outputs[0].shape, \
        'EfficientNet SE Block output shape mismatch'


test_efficientdet_model()
test_efficientdet_merge_outputs()
test_efficientnet_model()
test_efficientnet_se_block()
