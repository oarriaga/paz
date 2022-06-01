import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization
from tensorflow.keras.layers import Conv2D, SeparableConv2D, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from utils import get_drop_connect


def ClassNet(features, num_classes=90, num_anchors=9, num_filters=32,
             min_level=3, max_level=7, num_repeats=4, survival_rate=None,
             training=False, with_separable_conv=True, return_base=False,
             name='class_net/'):
    """Object class prediction network. Initialize the ClassNet.

    # Arguments
        features: List, feature to be processed by ClassNet head.
        num_classes: Integer. Number of classes.
        num_anchors: Integer. Number of anchors.
        num_filters: Integer. Number of filters for intermediate layers.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        num_repeats: Integer. Number of intermediate layers.
        survival_rate: Float.
        If a value is set then drop connect will be used.
        training: Bool, mode of using the network.
        with_separable_conv: Bool.
        True to use separable_conv instead of Conv2D.
        return_base: Bool.
        Build the base feature network only. Excluding final class head.
        name: String indicating the name of this layer.

    # Returns
        box_outputs: List, BoxNet output offsets for each level.
    """
    conv_blocks = []
    for repeat_args in range(num_repeats):
        conv_blocks.append(conv2d_layer(
            num_filters, 3, 'same', None, with_separable_conv,
            name + 'class-%d' % repeat_args, tf.zeros_initializer()))

    batchnorms = []
    for repeat_args in range(num_repeats):
        batchnorm_per_level = []
        for level in range(min_level, max_level + 1):
            batchnorm_per_level.append(BatchNormalization(
                name=name + 'class-%d-bn-%d' % (repeat_args, level)))
        batchnorms.append(batchnorm_per_level)

    classes = conv2d_layer(num_classes * num_anchors, 3, 'same', None,
                           with_separable_conv, name + 'class-predict',
                           tf.constant_initializer(-np.log((1 - 0.01) / 0.01)))

    class_outputs = []
    for level_id in range(0, max_level - min_level + 1):
        image = features[level_id]
        for repeat_args in range(num_repeats):
            original_image = image
            image = conv_blocks[repeat_args](image)
            image = batchnorms[repeat_args][level_id](image)
            image = tf.nn.swish(image)
            if repeat_args > 0 and survival_rate:
                image = get_drop_connect(image, training, survival_rate)
                image = image + original_image
        if return_base:
            class_outputs.append(image)
        else:
            class_outputs.append(classes(image))

    return class_outputs


def BoxNet(features, num_anchors=9, num_filters=32, min_level=3,
           max_level=7, num_repeats=4, survival_rate=None, training=False,
           with_separable_conv=True, return_base=False, name='box_net/'):
    """Initialize the BoxNet.

    # Arguments
        features: List, feature to be processed by BoxNet head.
        num_anchors: Integer. Number of anchors.
        num_filters: Integer. Number of filters for intermediate layers.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        num_repeats: Integer. Number of intermediate layers.
        survival_rate: Float.
        If a value is set then drop connect will be used.
        training: Bool, mode of using the network.
        with_separable_conv: Bool.
        True to use separable_conv instead of Conv2D.
        return_base: Bool.
        Build the base feature network only. Excluding final class head.
        name: String indicating the name of this layer.

    # Returns
        box_outputs: List, BoxNet output offsets for each level.
    """
    conv_blocks = []

    for repeat_args in range(num_repeats):
        conv_blocks.append(conv2d_layer(
            num_filters, 3, 'same', None, with_separable_conv,
            name + 'box-%d' % repeat_args, tf.zeros_initializer()))

    batchnorms = []
    for repeat_args in range(num_repeats):
        batchnorm_per_level = []
        for level in range(min_level, max_level + 1):
            batchnorm_per_level.append(BatchNormalization(
                name=name + 'box-%d-bn-%d' % (repeat_args, level)))
        batchnorms.append(batchnorm_per_level)

    boxes = conv2d_layer(4 * num_anchors, 3, 'same', None, with_separable_conv,
                         name + 'box-predict', tf.zeros_initializer())

    box_outputs = []
    for level_id in range(len(features)):
        image = features[level_id]
        for repeat_arg in range(num_repeats):
            original_image = image
            image = conv_blocks[repeat_arg](image)
            image = batchnorms[repeat_arg][level_id](image)
            image = tf.nn.swish(image)
            if repeat_arg > 0 and survival_rate:
                image = get_drop_connect(image, training, survival_rate)
                image = image + original_image
        if return_base:
            box_outputs.append(image)
        else:
            box_outputs.append(boxes(image))

    return box_outputs


def conv2d_layer(num_filters, kernel_size, padding, activation,
                 with_separable_conv, name, bias_initializer):
    """Gets the conv2d layer in ClassNet class.
    # Arguments
        num_filters: Integer. Number of intermediate layers.
        kernel_size: Integer. Kernel size of conv layer.
        padding: String. Padding for conv layer.
        activation: String. Activation function.
        with_separable_conv: Bool.
        True to use separable_conv instead of Conv2D.
        name: String. Name of conv layer.
        bias_initializer: String or TF Function. Bias initialization.

    # Returns
        conv2d_layer: TF conv layer.
    """
    if with_separable_conv:
        conv2d_layer = SeparableConv2D(
            num_filters, kernel_size, (1, 1), padding, 'channels_last',
            (1, 1), 1, activation, True,
            tf.initializers.variance_scaling(),
            tf.initializers.variance_scaling(), bias_initializer,
            name=name)
    else:
        conv2d_layer = Conv2D(
            num_filters, kernel_size, (1, 1), padding, 'channels_last',
            (1, 1), 1, activation, True,
            tf.random_normal_initializer(stddev=0.01),
            tf.zeros_initializer(), name=name)
    return conv2d_layer


def BiFPN(features, num_filters, id, fpn_weight_method):
    """
    BiFPN layer.
    # Arguments
    features: List, feature to be processed by BiFPN.
    num_filters: Integer. Number of filters for intermediate layers.
    id: Integer. Represents the BiFPN repetition count.
    fpn_weight_method: String representing the feature fusion method in BiFPN.

    # Returns
    features: List, features after BiFPN for the class and box heads.
    """
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5

        P6_in = Conv2D(
            num_filters, 1, 1, 'same', name='resample_p6/conv2d')(C5)
        P6_in = BatchNormalization(name='resample_p6/bn')(P6_in)
        P6_in = MaxPooling2D(3, 2, 'same', name='resample_p6/maxpool')(P6_in)
        P7_in = MaxPooling2D(3, 2, 'same', name='resample_p7/maxpool')(P6_in)

        P7_U = UpSampling2D()(P7_in)
        P6_td = FuseFeature(name=f'fpn_cells/cell_{id}/fnode0/add')(
            [P6_in, P7_U], fpn_weight_method)
        P6_td = tf.nn.swish(P6_td)
        P6_td = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5/conv')(P6_td)
        P6_td = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5/bn')(P6_td)

        P5_in_1 = Conv2D(
            num_filters, 1, 1, 'same',
            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)

        P6_U = UpSampling2D()(P6_td)
        P5_td = FuseFeature(name=f'fpn_cells/cell_{id}/fnode1/add')(
            [P5_in_1, P6_U], fpn_weight_method)
        P5_td = tf.nn.swish(P5_td)
        P5_td = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6/conv')(P5_td)
        P5_td = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6/bn')(P5_td)

        P4_in_1 = Conv2D(
            num_filters, 1, 1, 'same',
            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)

        P5_U = UpSampling2D()(P5_td)
        P4_td = FuseFeature(name=f'fpn_cells/cell_{id}/fnode2/add')(
            [P4_in_1, P5_U], fpn_weight_method)
        P4_td = tf.nn.swish(P4_td)
        P4_td = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7/conv')(P4_td)
        P4_td = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7/bn')(P4_td)

        P3_in = Conv2D(
            num_filters, 1, 1, 'same',
            name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)

        P4_U = UpSampling2D()(P4_td)
        P3_out = FuseFeature(name=f'fpn_cells/cell_{id}/fnode3/add')(
            [P3_in, P4_U], fpn_weight_method)
        P3_out = tf.nn.swish(P3_out)
        P3_out = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8/conv')(P3_out)
        P3_out = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8/bn')(P3_out)

        P4_in_2 = Conv2D(
            num_filters, 1, 1, 'same',
            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)

        P3_D = MaxPooling2D(3, 2, 'same')(P3_out)
        P4_out = FuseFeature(name=f'fpn_cells/cell_{id}/fnode4/add')(
            [P4_in_2, P4_td, P3_D], fpn_weight_method)
        P4_out = tf.nn.swish(P4_out)
        P4_out = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9/conv')(P4_out)
        P4_out = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9/bn')(P4_out)

        P5_in_2 = Conv2D(
            num_filters, 1, 1, 'same',
            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)

        P4_D = MaxPooling2D(3, 2, 'same')(P4_out)
        P5_out = FuseFeature(name=f'fpn_cells/cell_{id}/fnode5/add')(
            [P5_in_2, P5_td, P4_D], fpn_weight_method)
        P5_out = tf.nn.swish(P5_out)
        P5_out = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10/conv')(P5_out)
        P5_out = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10/bn')(P5_out)

        P5_D = MaxPooling2D(3, 2, 'same')(P5_out)
        P6_out = FuseFeature(name=f'fpn_cells/cell_{id}/fnode6/add')(
            [P6_in, P6_td, P5_D], fpn_weight_method)
        P6_out = tf.nn.swish(P6_out)
        P6_out = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11/conv')(P6_out)
        P6_out = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11/bn')(P6_out)

        P6_D = MaxPooling2D(3, 2, 'same')(P6_out)
        P7_out = FuseFeature(name=f'fpn_cells/cell_{id}/fnode7/add')(
            [P7_in, P6_D], fpn_weight_method)
        P7_out = tf.nn.swish(P7_out)
        P7_out = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12/conv')(P7_out)
        P7_out = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12/bn')(P7_out)

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features

        P7_U = UpSampling2D()(P7_in)
        P6_td = FuseFeature(name=f'fpn_cells/cell_{id}/fnode0/add')(
            [P6_in, P7_U], fpn_weight_method)
        P6_td = tf.nn.swish(P6_td)
        P6_td = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5/conv')(P6_td)
        P6_td = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5/bn')(P6_td)

        P6_U = UpSampling2D()(P6_td)
        P5_td = FuseFeature(name=f'fpn_cells/cell_{id}/fnode1/add')(
            [P5_in, P6_U], fpn_weight_method)
        P5_td = tf.nn.swish(P5_td)
        P5_td = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6/conv')(P5_td)
        P5_td = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6/bn')(P5_td)

        P5_U = UpSampling2D()(P5_td)
        P4_td = FuseFeature(name=f'fpn_cells/cell_{id}/fnode2/add')(
            [P4_in, P5_U], fpn_weight_method)
        P4_td = tf.nn.swish(P4_td)
        P4_td = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7/conv')(P4_td)
        P4_td = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7/bn')(P4_td)

        P4_U = UpSampling2D()(P4_td)
        P3_out = FuseFeature(name=f'fpn_cells/cell_{id}/fnode3/add')(
            [P3_in, P4_U], fpn_weight_method)
        P3_out = tf.nn.swish(P3_out)
        P3_out = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8/conv')(P3_out)
        P3_out = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8/bn')(P3_out)

        P3_D = MaxPooling2D(3, 2, 'same')(P3_out)
        P4_out = FuseFeature(name=f'fpn_cells/cell_{id}/fnode4/add')(
            [P4_in, P4_td, P3_D], fpn_weight_method)
        P4_out = tf.nn.swish(P4_out)
        P4_out = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9/conv')(P4_out)
        P4_out = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9/bn')(P4_out)

        P4_D = MaxPooling2D(3, 2, 'same')(P4_out)
        P5_out = FuseFeature(name=f'fpn_cells/cell_{id}/fnode5/add')(
            [P5_in, P5_td, P4_D], fpn_weight_method)
        P5_out = tf.nn.swish(P5_out)
        P5_out = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10/conv')(P5_out)
        P5_out = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10/bn')(P5_out)

        P5_D = MaxPooling2D(3, 2, 'same')(P5_out)
        P6_out = FuseFeature(name=f'fpn_cells/cell_{id}/fnode6/add')(
            [P6_in, P6_td, P5_D], fpn_weight_method)
        P6_out = tf.nn.swish(P6_out)
        P6_out = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11/conv')(P6_out)
        P6_out = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11/bn')(P6_out)

        P6_D = MaxPooling2D(3, 2, 'same')(P6_out)
        P7_out = FuseFeature(name=f'fpn_cells/cell_{id}/fnode7/add')(
            [P7_in, P6_D], fpn_weight_method)
        P7_out = tf.nn.swish(P7_out)
        P7_out = SeparableConv2D(
            num_filters, 3, 1, 'same', use_bias=True,
            name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12/conv')(P7_out)
        P7_out = BatchNormalization(
            name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12/bn')(P7_out)

    return P3_out, P4_out, P5_out, P6_out, P7_out


class FuseFeature(Layer):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(self.name, (num_in,), tf.float32,
                                 tf.keras.initializers.constant(1 / num_in),
                                 trainable=True)

    def call(self, inputs, fpn_weight_method):
        """
        # Arguments
        inputs: Tensor. Features to be fused.
        fpn_weight_method: String representing the feature fusion method.

        # Returns
        x: feature after combining by the feature fusion method in BiFPN.
        """
        if fpn_weight_method == 'fastattention':
            w = tf.keras.activations.relu(self.w)
            x = tf.reduce_sum(
                [w[i] * inputs[i] for i in range(len(inputs))], 0)
            x = x / (tf.reduce_sum(w) + 0.0001)
        elif fpn_weight_method == 'sum':
            x = inputs[0]
            for node in inputs[1:]:
                x = x + node
        else:
            raise ValueError('FPN weight fusion is not defined')
        return x
