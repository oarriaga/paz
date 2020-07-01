from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from ..layers import Conv2DNormalization

import numpy as np
from itertools import product


def create_multibox_head(tensors, num_classes, num_priors, l2_loss=0.0005,
                         num_regressions=4, l2_norm=False, batch_norm=False):
    """Adds multibox head with classification and regression output tensors.

    # Arguments
        tensors: List of tensors.
        num_classes: Int. Number of classes.
        num_priors. List of integers. Length should equal to tensors length.
            Each integer represents the amount of bounding boxes shapes in
            each feature map value.
        l2_loss: Float. L2 loss value to be added to convolutional layers.
        num_regressions: Number of values to be regressed per prior box.
            e.g. for 2D bounding boxes we regress 4 coordinates.
        l2_norm: Boolean. If `True` l2 normalization layer is applied to
            each before a convolutional layer.
        batch_norm: Boolean. If `True` batch normalization is applied after
            each convolutional layer.
    """
    classification_layers, regression_layers = [], []
    for layer_arg, base_layer in enumerate(tensors):
        if l2_norm:
            base_layer = Conv2DNormalization(20)(base_layer)

        # classification leaf -------------------------------------------------
        num_kernels = num_priors[layer_arg] * num_classes
        class_leaf = Conv2D(num_kernels, 3, padding='same',
                            kernel_regularizer=l2(l2_loss))(base_layer)
        if batch_norm:
            class_leaf = BatchNormalization()(class_leaf)
        class_leaf = Flatten()(class_leaf)
        classification_layers.append(class_leaf)

        # regression leaf -----------------------------------------------------
        num_kernels = num_priors[layer_arg] * num_regressions
        regress_leaf = Conv2D(num_kernels, 3, padding='same',
                              kernel_regularizer=l2(l2_loss))(base_layer)
        if batch_norm:
            regress_leaf = BatchNormalization()(regress_leaf)

        regress_leaf = Flatten()(regress_leaf)
        regression_layers.append(regress_leaf)

    classifications = Concatenate(axis=1)(classification_layers)
    regressions = Concatenate(axis=1)(regression_layers)
    num_boxes = K.int_shape(regressions)[-1] // num_regressions
    classifications = Reshape((num_boxes, num_classes))(classifications)
    classifications = Activation('softmax')(classifications)
    regressions = Reshape((num_boxes, num_regressions))(regressions)
    outputs = Concatenate(
        axis=2, name='boxes')([regressions, classifications])
    return outputs


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
