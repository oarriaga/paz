from keras.regularizers import l2
from keras.layers import (
    Conv2D,
    Activation,
    BatchNormalization,
    Flatten,
    Reshape,
    Concatenate,
)
import numpy as np
from ..layers import Conv2DNormalization


def build_multibox_head(
    tensors,
    num_classes,
    num_priors,
    l2_loss=0.0005,
    num_regressions=4,
    l2_norm=False,
    batch_norm=False,
):
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
        class_leaf = Conv2D(
            num_kernels, 3, padding="same", kernel_regularizer=l2(l2_loss)
        )(base_layer)
        if batch_norm:
            class_leaf = BatchNormalization()(class_leaf)
        class_leaf = Flatten()(class_leaf)
        classification_layers.append(class_leaf)

        # regression leaf -----------------------------------------------------
        num_kernels = num_priors[layer_arg] * num_regressions
        regress_leaf = Conv2D(
            num_kernels, 3, padding="same", kernel_regularizer=l2(l2_loss)
        )(base_layer)
        if batch_norm:
            regress_leaf = BatchNormalization()(regress_leaf)

        regress_leaf = Flatten()(regress_leaf)
        regression_layers.append(regress_leaf)

    classifications = Concatenate(axis=1)(classification_layers)
    regressions = Concatenate(axis=1)(regression_layers)
    num_boxes = np.shape(regressions)[-1] // num_regressions
    classifications = Reshape((num_boxes, num_classes))(classifications)
    classifications = Activation("softmax")(classifications)
    regressions = Reshape((num_boxes, num_regressions))(regressions)
    outputs = Concatenate(axis=2, name="boxes")([regressions, classifications])
    return outputs
