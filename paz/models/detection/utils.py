from __future__ import division
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from ..layers import Conv2DNormalization

import numpy as np
from itertools import product


def create_multibox_head(
        output_tensors, num_classes, num_priors, with_l2_norm=False,
        with_batch_norm=False, l2_regularization=0.0005, num_regressions=4,
        base_name=''):

    classification_layers, regression_layers = [], []
    for layer_arg, base_layer in enumerate(output_tensors):

        if with_l2_norm:
            base_layer = Conv2DNormalization(20)(base_layer)
        str_arg = str(layer_arg)

        # classification leaf
        class_name = 'classification_leaf_' + str(layer_arg)
        class_leaf = Conv2D(
            num_priors[layer_arg] * num_classes, (3, 3),
            padding='same', name=base_name + class_name,
            kernel_regularizer=l2(l2_regularization))(base_layer)

        if with_batch_norm:
            class_leaf = BatchNormalization(
                name=base_name + 'batch_norm_ssd_3_' + str_arg)(class_leaf)

        class_leaf = Flatten(
            name=base_name + 'flat_classification_' + str_arg)(class_leaf)
        classification_layers.append(class_leaf)

        # regression leaf
        regress_name = 'regression_leaf_' + str(layer_arg)
        regress_leaf = Conv2D(
            num_priors[layer_arg] * num_regressions, (3, 3),
            padding='same', name=base_name + regress_name,
            kernel_regularizer=l2(l2_regularization))(base_layer)

        if with_batch_norm:
            regress_leaf = BatchNormalization(
                name=base_name + 'batch_norm_ssd_4_' + str_arg)(regress_leaf)

        regress_leaf = Flatten(
            name=base_name + 'flat_regression_' + str_arg)(regress_leaf)
        regression_layers.append(regress_leaf)

    classifications = concatenate(
        classification_layers, axis=1,
        name=base_name + 'concatenate_classification_' + str_arg)
    regressions = concatenate(
        regression_layers, axis=1,
        name=base_name + 'concatenate_regression_' + str_arg)

    if hasattr(regressions, '_keras_shape'):
        num_boxes = regressions._keras_shape[-1] // num_regressions
        print('_kersa_shape', num_boxes)
    elif hasattr(regressions, 'int_shape'):
        num_boxes = K.int_shape(regressions)[-1] // num_regressions
        print('_int_shape', num_boxes)
    num_boxes = K.int_shape(regressions)[-1] // num_regressions
    print(num_boxes)
    classifications = Reshape(
        (num_boxes, num_classes),
        name=base_name + 'reshape_classification' + str_arg)(classifications)

    classifications = Activation(
        'softmax',
        name=base_name + 'softmax_activation_' + str_arg)(classifications)

    regressions = Reshape(
        (num_boxes, num_regressions),
        name=base_name + 'reshape_regression_' + str_arg)(regressions)

    # super pose hack
    if num_regressions == 8:
        box_reg, pose_reg = Lambda(split_regressions)(regressions)
        pose_reg = Activation('tanh')(pose_reg)
        regressions = Concatenate(axis=2)([box_reg, pose_reg])

    boxes_output = concatenate([regressions, classifications],
                               axis=2, name='predictions')
    return boxes_output


def split_regressions(x):
    return [x[:, :, :4], x[:, :, 4:8]]


def create_prior_boxes(configuration_name='VOC'):
    configuration = get_prior_box_configuration(configuration_name)
    image_size = configuration['image_size']
    feature_map_sizes = configuration['feature_map_sizes']
    min_sizes = configuration['min_sizes']
    max_sizes = configuration['max_sizes']
    steps = configuration['steps']
    model_aspect_ratios = configuration['aspect_ratios']
    mean = []
    for feature_map_arg, feature_map_size in enumerate(feature_map_sizes):
        step = steps[feature_map_arg]
        min_size = min_sizes[feature_map_arg]
        max_size = max_sizes[feature_map_arg]
        aspect_ratios = model_aspect_ratios[feature_map_arg]
        for y, x in product(range(feature_map_size), repeat=2):
            f_k = image_size / step
            center_x = (x + 0.5) / f_k
            center_y = (y + 0.5) / f_k
            s_k = min_size / image_size
            mean = mean + [center_x, center_y, s_k, s_k]
            s_k_prime = np.sqrt(s_k * (max_size / image_size))
            mean = mean + [center_x, center_y, s_k_prime, s_k_prime]
            for aspect_ratio in aspect_ratios:
                mean = mean + [center_x, center_y, s_k * np.sqrt(aspect_ratio),
                               s_k / np.sqrt(aspect_ratio)]
                mean = mean + [center_x, center_y, s_k / np.sqrt(aspect_ratio),
                               s_k * np.sqrt(aspect_ratio)]

    output = np.asarray(mean).reshape((-1, 4))
    # output = np.clip(output, 0, 1)
    return output


def get_prior_box_configuration(configuration_name='VOC'):
    if configuration_name in {'VOC', 'FAT'}:
        configuration = {
            'feature_map_sizes': [38, 19, 10, 5, 3, 1],
            'image_size': 300,
            'steps': [8, 16, 32, 64, 100, 300],
            'min_sizes': [30, 60, 111, 162, 213, 264],
            'max_sizes': [60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            'variance': [0.1, 0.2]}

    elif configuration_name in {'COCO', 'YCBVideo'}:
        configuration = {
            'feature_map_sizes': [64, 32, 16, 8, 4, 2, 1],
            'image_size': 512,
            'steps': [8, 16, 32, 64, 128, 256, 512],
            'min_sizes': [21, 51, 133, 215, 297, 379, 461],
            'max_sizes': [51, 133, 215, 297, 379, 461, 542],
            'aspect_ratios': [[2], [2, 3], [2, 3],
                              [2, 3], [2, 3], [2], [2]],
            'variance': [0.1, 0.2]}
    else:
        raise ValueError('Invalid configuration name:', configuration_name)
    return configuration
