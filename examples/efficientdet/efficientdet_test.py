from PIL import Image
import numpy as np
import tensorflow as tf
from efficientdet_model import EfficientDet
from config import get_efficientdet_default_params


def efficient_model_configuration():
    config = dict()
    config['model_name'] = 'efficientdet-d0'
    config['backbone_name'] = 'efficientnet-b0'
    config['backbone_weight'] = 'imagenet'
    config['act_type'] = 'swish'
    config['min_level'] = 3
    config['max_level'] = 7
    config['fpn_name'] = 'BiFPN'
    config['fpn_weight_method'] = 'fastattn'
    config['fpn_num_filters'] = 64
    config['fpn_cell_repeats'] = 3
    config['use_batchnorm_for_sampling'] = True
    config['conv_after_downsample'] = False
    config['conv_batchnorm_act_pattern'] = False
    config['separable_conv'] = True
    config['aspect_ratios'] = [1.0, 2.0, 0.5]
    config['survival_prob'] = None
    config['num_classes'] = 90
    config['num_scales'] = 3
    config['anchor_scale'] = 4
    config['image_size'] = 512
    config['box_class_repeats'] = 3
    config['feature_only'] = False
    return config


def get_test_image():
    image = tf.zeros((1, 512, 512, 3), dtype=tf.float32)
    return image


def assert_model_outputs(detector, image, config):
    class_out, det_out = detector(image)
    assert len(class_out) == (config['max_level'] - config['min_level'] + 1),\
        'Class outputs length fail'
    assert len(det_out) == (config['max_level'] - config['min_level'] + 1), \
        'Class outputs length fail'


def test_efficientdet_model():
    config = efficient_model_configuration()
    image = get_test_image()
    detector = EfficientDet()
    assert_model_outputs(detector,
                         image,
                         config)


def test_efficientdet_default_params():
    param_names = ['backbone_name', 'image_size',
                   'fpn_num_filters', 'fpn_cell_repeats',
                   'box_class_repeats', 'anchor_scale',
                   'min_level', 'max_level', 'fpn_weight_method'
                   ]
    params = get_efficientdet_default_params('efficientdet-d5')
    assert list(params.keys()) == param_names


test_efficientdet_model()
test_efficientdet_default_params()
