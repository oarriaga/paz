import tensorflow as tf
from efficientdet_model import EfficientDet
from config import get_efficientdet_default_params
from postprocess import merge_class_box_level_outputs


def get_test_image(image_size):
    image = tf.zeros((1, image_size, image_size, 3), dtype=tf.float32)
    return image


def test_efficientdet_model():
    detector = EfficientDet(model_name='efficientdet-d0')
    params = get_efficientdet_default_params('efficientdet-d0')
    image_size = params['image_size']
    num_levels = params['max_level'] - params['min_level'] + 1
    image = get_test_image(image_size)
    class_outs, det_outs = detector(image)
    assert len(class_outs) == num_levels, 'Class outputs length fail'
    assert len(det_outs) == num_levels, 'Box outputs length fail'


def test_efficientdet_default_params():
    param_names = ['backbone_name', 'image_size',
                   'fpn_num_filters', 'fpn_cell_repeats',
                   'box_class_repeats', 'anchor_scale',
                   'min_level', 'max_level', 'fpn_weight_method'
                   ]
    params = get_efficientdet_default_params('efficientdet-d5')
    assert list(params.keys()) == param_names, 'Default params not matching'


def test_efficientdet_merge_outputs():
    detector = EfficientDet(model_name='efficientdet-d0')
    params = get_efficientdet_default_params('efficientdet-d0')
    image_size = params['image_size']
    num_levels = params['max_level'] - params['min_level'] + 1
    num_classes = 90
    image = get_test_image(image_size)
    class_outs, box_outs = detector(image)
    class_outs_expected = tf.zeros((1, 49104, 90), dtype=tf.float32)
    box_outs_expected = tf.zeros((1, 49104, 4), dtype=tf.float32)
    class_outs, box_outs = merge_class_box_level_outputs(class_outs,
                                                         box_outs,
                                                         num_levels,
                                                         num_classes)
    assert class_outs_expected.shape == class_outs.shape, \
        'Merged class outputs not matching'
    assert box_outs_expected.shape == box_outs.shape,\
        'Merged box outputs not matching'


test_efficientdet_model()
test_efficientdet_default_params()
test_efficientdet_merge_outputs()
