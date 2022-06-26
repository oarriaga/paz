import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Layer,
                                     MaxPooling2D, SeparableConv2D,
                                     UpSampling2D)

from utils import get_drop_connect


def ClassNet(features, num_classes=90, num_anchors=9, num_filters=32,
             min_level=3, max_level=7, repeats=4, survival_rate=None,
             training=False, with_separable_conv=True, return_base=False,
             name='class_net/'):
    """Object class prediction network. Initialize the ClassNet.

    # Arguments
        features: List, feature to be processed by ClassNet head.
        num_classes: Integer. Number of classes.
        num_anchors: Integer. Number of anchors.
        num_filters: Integer. Number of filters for intermediate
                     layers.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        repeats: Integer. Number of intermediate layers.
        survival_rate: Float.
        If a value is set then drop connect will be used.
        training: Bool, mode of using the network.
        with_separable_conv: Bool.
        True to use separable_conv instead of Conv2D.
        return_base: Bool.
        Build the base feature network only. Excluding final
        class head.
        name: String indicating the name of this layer.

    # Returns
        box_outputs: List, BoxNet output offsets for each level.
    """

    class_outputs = build_predictionnet(
        repeats, num_filters, with_separable_conv, name, min_level,
        max_level, num_classes, num_anchors, features, survival_rate,
        training, return_base, True)
    return class_outputs


def BoxNet(features, num_anchors=9, num_filters=32, min_level=3,
           max_level=7, repeats=4, survival_rate=None, training=False,
           with_separable_conv=True, return_base=False,
           name='box_net/'):
    """Initialize the BoxNet.

    # Arguments
        features: List, feature to be processed by BoxNet head.
        num_anchors: Integer. Number of anchors.
        num_filters: Integer. Number of filters for intermediate
                     layers.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        repeats: Integer. Number of intermediate layers.
        survival_rate: Float.
        If a value is set then drop connect will be used.
        training: Bool, mode of using the network.
        with_separable_conv: Bool.
        True to use separable_conv instead of Conv2D.
        return_base: Bool.
        Build the base feature network only. Excluding final
        class head.
        name: String indicating the name of this layer.

    # Returns
        box_outputs: List, BoxNet output offsets for each level.
    """

    box_outputs = build_predictionnet(
        repeats, num_filters, with_separable_conv, name, min_level,
        max_level, None, num_anchors, features, survival_rate,
        training, return_base, False)
    return box_outputs


def BiFPN(features, num_filters, id, fpn_weight_method):
    """
    BiFPN layer.
    # Arguments
    features: List, feature to be processed by BiFPN.
    num_filters: Integer. Number of filters for intermediate layers.
    id: Integer. Represents the BiFPN repetition count.
    fpn_weight_method: String representing the feature fusion method
                       in BiFPN.

    # Returns
    features: List, features after BiFPN for the class and box heads.
    """
    is_non_repeated_block = id == 0

    if is_non_repeated_block:
        P6_in, P7_in = preprocess_features_BiFPN(
            0, features[-1], num_filters, features, id, True)

        previous_layer_feature, current_feature = P7_in, P6_in
        feature_tds = propagate_downwards_BiFPN(
            is_non_repeated_block, features, previous_layer_feature,
            current_feature, fpn_weight_method, num_filters, id)

        current_layer_feature, next_input = feature_tds[3], features[-2]
        next_td, output_features = feature_tds[2], [feature_tds[3]]
        output_features = propagate_upwards_BiFPN(
            is_non_repeated_block, features, current_layer_feature,
            next_input, next_td, id, feature_tds, fpn_weight_method,
            num_filters, output_features, P6_in, P7_in)

        P3_out, P4_out, P5_out, P6_out, P7_out = output_features

    else:
        previous_layer_feature, current_feature = features[-1], features[-2]
        feature_tds = propagate_downwards_BiFPN(
            is_non_repeated_block, features, previous_layer_feature,
            current_feature, fpn_weight_method, num_filters, id)

        current_layer_feature, next_input = feature_tds[-1], features[1]
        next_td, output_features = feature_tds[-2], [feature_tds[-1]]
        output_features = propagate_upwards_BiFPN(
            is_non_repeated_block, features, current_layer_feature,
            next_input, next_td, id, feature_tds, fpn_weight_method,
            num_filters, output_features, None, None)

        P3_out, P4_out, P5_out, P6_out, P7_out = output_features

    return P3_out, P4_out, P5_out, P6_out, P7_out


class FuseFeature(Layer):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(
            self.name, (num_in,), tf.float32,
            tf.keras.initializers.constant(1 / num_in), trainable=True)

    def call(self, inputs, fpn_weight_method):
        """
        # Arguments
        inputs: Tensor. Features to be fused.
        fpn_weight_method: String representing the feature fusion
                           method.

        # Returns
        x: feature after combining by the feature fusion method in
           BiFPN.
        """
        if fpn_weight_method == 'fastattention':
            w = tf.keras.activations.relu(self.w)

            pre_activations = []
            for input_idx in range(len(inputs)):
                pre_activations.append(w[input_idx] * inputs[input_idx])
            x = tf.reduce_sum(pre_activations, 0)
            x = x / (tf.reduce_sum(w) + 0.0001)
        elif fpn_weight_method == 'sum':
            x = inputs[0]
            for node in inputs[1:]:
                x = x + node
        else:
            raise ValueError('FPN weight fusion is not defined')
        return x


def conv2D_layer(num_filters, kernel_size, padding, activation,
                 with_separable_conv, name, bias_initializer):
    """Gets the conv2D layer in ClassNet class.
    # Arguments
        num_filters: Integer. Number of intermediate layers.
        kernel_size: Integer. Kernel size of conv layer.
        padding: String. Padding for conv layer.
        activation: String. Activation function.
        with_separable_conv: Bool.
        True to use separable_conv instead of Conv2D.
        name: String. Name of conv layer.
        bias_initializer: String or TF Function. Bias
                          initialization.

    # Returns
        conv2D_layer: TF conv layer.
    """
    if with_separable_conv:
        conv2D_layer = SeparableConv2D(
            num_filters, kernel_size, (1, 1), padding, 'channels_last',
            (1, 1), 1, activation, True, tf.initializers.variance_scaling(),
            tf.initializers.variance_scaling(), bias_initializer, name=name)
    else:
        conv2D_layer = Conv2D(
            num_filters, kernel_size, (1, 1), padding, 'channels_last',
            (1, 1), 1, activation, True,
            tf.random_normal_initializer(stddev=0.01),
            tf.zeros_initializer(), name=name)
    return conv2D_layer


def build_predictionnet(repeats, num_filters, with_separable_conv, name,
                        min_level, max_level, num_classes, num_anchors,
                        features, survival_rate, training,
                        return_base, build_classnet):
    conv_blocks = build_predictionnet_conv_blocks(
        repeats, num_filters, with_separable_conv, name, build_classnet)

    batchnorms = build_predictionnet_batchnorm_blocks(
        repeats, min_level, max_level, name, build_classnet)

    if build_classnet:
        bias_initializer = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        num_filters = num_classes * num_anchors
        num_levels = max_level - min_level + 1
        layer_name = name + 'class-predict'
    else:
        bias_initializer = tf.zeros_initializer()
        num_filters = 4 * num_anchors
        num_levels = len(features)
        layer_name = name + 'box-predict'

    classes = conv2D_layer(
        num_filters, 3, 'same', None, with_separable_conv, layer_name,
        bias_initializer)

    predictor_outputs = []
    for level_id in range(num_levels):
        image = propagate_forward_predictionnet(
            features, level_id, repeats, conv_blocks, batchnorms,
            survival_rate, training, return_base, classes)
        predictor_outputs.append(image)
    return predictor_outputs


def build_predictionnet_conv_blocks(repeats, num_filters, with_separable_conv,
                                    name, is_classnet):

    layer_name_prefix = name + 'class' if is_classnet else name + 'box'
    conv_blocks = []
    for repeat_args in range(repeats):
        conv_blocks.append(conv2D_layer(
            num_filters, 3, 'same', None, with_separable_conv,
            layer_name_prefix + '-%d' % repeat_args, tf.zeros_initializer()))
    return conv_blocks


def build_predictionnet_batchnorm_blocks(repeats, min_level, max_level,
                                         name, is_classnet):

    layer_name_prefix = name + 'class' if is_classnet else name + 'box'
    batchnorms = []
    for repeat_args in range(repeats):
        batchnorm_per_level = []
        for level in range(min_level, max_level + 1):
            batchnorm_per_level.append(BatchNormalization(
                name=layer_name_prefix + '-%d-bn-%d' % (repeat_args, level)))
        batchnorms.append(batchnorm_per_level)
    return batchnorms


def propagate_forward_predictionnet(features, level_id, repeats, conv_blocks,
                                    batchnorms, survival_rate, training,
                                    return_base, output_candidates):
    image = features[level_id]
    for repeat_args in range(repeats):
        original_image = image
        image = conv_blocks[repeat_args](image)
        image = batchnorms[repeat_args][level_id](image)
        image = tf.nn.swish(image)
        if repeat_args > 0 and survival_rate:
            image = get_drop_connect(image, training, survival_rate)
            image = image + original_image

    if return_base:
        return image
    else:
        return output_candidates(image)


def propagate_downwards_BiFPN(is_non_repeated_block, features,
                              previous_layer_feature, current_feature,
                              fpn_weight_method, num_filters, id):

    feature_tds = [] if is_non_repeated_block else [previous_layer_feature]
    for depth_idx in range(len(features)-1):
        previous_layer_feature = propagate_downwards(
            previous_layer_feature, current_feature, id,
            fpn_weight_method, depth_idx, features, num_filters)
        current_feature_idx = -1 - depth_idx if is_non_repeated_block\
            else len(features) - 3 - depth_idx
        current_feature = features[current_feature_idx]
        feature_tds.append(previous_layer_feature)
    return feature_tds


def propagate_downwards(previous_layer_feature, current_feature, id,
                        fpn_weight_method, depth_idx, features,
                        num_filters):
    """Propagates features in downward direction starting from the
    features of top most layer of EfficientNet backbone.

    # Arguments
        previous_layer_feature :Tensor, feature from the relatively
                                top layer.
        current_feature :Tensor, feature from the current layer.
        id :Int, the ID or index of the BiFPN block.
        fpn_weight_method :string, String representing the feature
                           fusion method.
        depth_idx :Int, the depth of the feature of BiFPN layer.
        features :List, the features returned from EfficientNet
                  backbone.
        num_filters :Int, Number of filters for intermediate layers.

    # Returns
        current_feature_td: Tensor, tensor resulting from
                            down propagation in BiFPN layer.
    """
    is_non_repeated_block = id == 0
    layer_not_P7 = depth_idx > 0
    if is_non_repeated_block and layer_not_P7:
        current_feature = preprocess_features_BiFPN(
            depth_idx, current_feature, num_filters, features, id, True)

    previous_layer_feature_U = UpSampling2D()(previous_layer_feature)
    current_feature_td = FuseFeature(
        name=(f'fpn_cells/cell_{id}/fnode{depth_idx}/add'))(
        [current_feature, previous_layer_feature_U], fpn_weight_method)

    current_feature_td = tf.nn.swish(current_feature_td)
    current_feature_td = SeparableConv2D(
        num_filters, 3, 1, 'same', use_bias=True,
        name=(f'fpn_cells/cell_{id}/fnode{depth_idx}/'
              f'op_after_combine{len(features) + depth_idx}/conv'))(
            current_feature_td)

    current_feature_td = BatchNormalization(
        name=(f'fpn_cells/cell_{id}/fnode{depth_idx}/'
              f'op_after_combine{len(features) + depth_idx}/bn'))(
            current_feature_td)
    return current_feature_td


def propagate_upwards_BiFPN(is_non_repeated_block, features,
                            current_layer_feature, next_input, next_td,
                            id, feature_tds, fpn_weight_method,
                            num_filters, output_features, P6_in, P7_in):

    for depth_idx in range(len(features) - 1):
        current_layer_feature = propagate_upwards(
            current_layer_feature, next_input, next_td, id, feature_tds,
            depth_idx, fpn_weight_method, num_filters, features)
        output_features.append(current_layer_feature)

        depth_idx_arg, P6_in_arg, P7_in_arg, next_input_arg, next_td_arg =\
            (depth_idx + 1, P6_in, P7_in, None, None) if is_non_repeated_block\
            else (depth_idx, None, None, next_input, next_td)
        next_input, next_td = compute_next_input_feature_BiFPN(
            id, features, feature_tds, depth_idx_arg, P6_in_arg,
            P7_in_arg, next_input_arg, next_td_arg)
    return output_features


def propagate_upwards(current_layer_feature, next_input, next_td, id,
                      feature_tds, depth_idx, fpn_weight_method,
                      num_filters, features):
    """Propagates features in upward direction starting from the
    features of bottom most layer of EfficientNet backbone.

    # Arguments
        current_layer_feature :Tensor, Tensor, feature from the
                               current layer.
        next_input :Tensor, Tensor, feature from the relatively
                    top layer.
        next_td : Tensor, The feature tensor from the relatively
                    top layer as result of upward or downward
                    propagation.
        id :Int, the ID or index of the BiFPN block.
        feature_tds: List, the list of features as a result of
                     upward or downward propagation.
        depth_idx :Int, the depth of the feature of BiFPN layer.
        fpn_weight_method :string, String representing the feature
                           fusion method.
        num_filters :Int, Number of filters for intermediate layers.
        features :List, the features returned from EfficientNet
                  backbone.

    # Returns
        current_feature_td :Tensor, Tensor, tensor resulting from
                            upward propagation in BiFPN layer.
    """
    current_layer_feature_D = MaxPooling2D(3, 2, 'same')(current_layer_feature)

    is_non_repeated_block = id == 0
    is_layer_P6_or_P7 = depth_idx < 2
    is_layer_P4 = depth_idx == 3

    if is_non_repeated_block:
        if is_layer_P6_or_P7:
            next_input = preprocess_features_BiFPN(
                depth_idx, next_input, num_filters, features, id, False)

        layer_names = [(f'fpn_cells/cell_{id}/fnode'
                        f'{len(features) - 2 + depth_idx + 1}'
                        f'/add'),
                       (f'fpn_cells/cell_{id}/fnode'
                        f'{len(feature_tds) + depth_idx}'
                        f'/op_after_combine{9 + depth_idx}'
                        f'/conv'),
                       (f'fpn_cells/cell_{id}/fnode'
                        f'{len(feature_tds) + depth_idx}/'
                        f'op_after_combine{9 + depth_idx}'
                        f'/bn')]
    else:
        layer_names = [(f'fpn_cells/cell_{id}/'
                        f'fnode{len(feature_tds) - 1 + depth_idx}'
                        f'/add'),
                       (f'fpn_cells/cell_{id}/fnode'
                        f'{len(feature_tds) - 1 + depth_idx}'
                        f'/op_after_combine'
                        f'%d' % (len(feature_tds) + depth_idx + len(features)
                                 - 2 + 1) + '/conv'),
                       (f'fpn_cells/cell_{id}/fnode'
                        f'{len(feature_tds) - 1 + depth_idx}/'
                        f'op_after_combine'
                        f'%d' % (len(feature_tds) + depth_idx + len(features)
                                 - 2 + 1) + '/bn')]

    to_fuse = [next_input, current_layer_feature_D] if is_layer_P4\
        else [next_input, next_td, current_layer_feature_D]

    next_out = FuseFeature(name=layer_names[0])(
                           to_fuse, fpn_weight_method)
    next_out = tf.nn.swish(next_out)
    next_out = SeparableConv2D(num_filters, 3, 1, 'same', use_bias=True,
                               name=layer_names[1])(next_out)
    next_out = BatchNormalization(name=layer_names[2])(next_out)
    return next_out


def preprocess_features_BiFPN(depth_idx, input_feature, num_filters,
                              features, id, is_propagate_downwards):
    """Perform pre-processing on features before applying
    downward propagation or upward propagation.

    # Arguments
        depth_idx :Int, the depth of the feature of BiFPN layer.
        input_feature :Tensor, feature from the current layer.
        num_filters :Int, Number of filters for intermediate layers.
        features :List, the features returned from EfficientNet
                  backbone.
        id :Int, the ID or index of the BiFPN block.
        is_propagate_downwards :Bool, Boolean flag indicating if
                                propagation is in upward or
                                downward direction.

    # Returns
        preprocessed_feature: Tensor, the preprocessed feature.
    """
    is_layer_P7 = depth_idx == 0

    if is_propagate_downwards:
        if is_layer_P7:
            layer_names = [(f'resample_p{len(features) + 1}/conv2d'),
                           (f'resample_p{len(features) + 1}/bn')]
            P6_in = preprocess_features_partly_BiFPN(
                input_feature, num_filters, layer_names)
            P6_in = MaxPooling2D(3, 2, 'same', name=(f'resample_p'
                                                     f'{len(features) + 1}'
                                                     f'/maxpool'))(P6_in)
            P7_in = MaxPooling2D(3, 2, 'same', name=(f'resample_p'
                                                     f'{len(features) + 2}'
                                                     f'/maxpool'))(P6_in)
            return P6_in, P7_in
        else:
            layer_names = [(f'fpn_cells/cell_{id}/fnode{depth_idx}/resample_0_'
                            f'{3 - depth_idx}_{len(features) + depth_idx}'
                            f'/conv2d'),
                           (f'fpn_cells/cell_{id}/fnode{depth_idx}/resample_0_'
                            f'{3 - depth_idx}_{len(features) + depth_idx}/bn')]
            preprocessed_feature = preprocess_features_partly_BiFPN(
                input_feature, num_filters, layer_names)
            return preprocessed_feature
    else:
        layer_names = [(f'fpn_cells/cell_{id}/fnode' +
                        '%d' % (len(features) - 1 + depth_idx) +
                        f'/resample_0_{1 + depth_idx}_{9 + depth_idx}/conv2d'),
                       (f'fpn_cells/cell_{id}'
                        f'/fnode{len(features) - 1 + depth_idx}'
                        f'/resample_0_{1 + depth_idx}_{9 + depth_idx}/bn')]
        preprocessed_feature = preprocess_features_partly_BiFPN(
            input_feature, num_filters, layer_names)
        return preprocessed_feature


def preprocess_features_partly_BiFPN(input_feature, num_filters, layer_names):
    """Perform a part of feature preprocessing such as
    applying Conv2D and BatchNormalization.

    # Arguments
        input_feature :Tensor, feature from the current layer.
        num_filters :Int, Number of filters for intermediate layers.
        layer_names :List, name of the layers.

    # Returns
        partly_processed_feature: Tensor, the partly preprocessed
                                  feature.
    """
    partly_processed_feature = Conv2D(
        num_filters, 1, 1, 'same', name=layer_names[0])(input_feature)
    partly_processed_feature = BatchNormalization(
        name=layer_names[1])(partly_processed_feature)
    return partly_processed_feature


def compute_next_input_feature_BiFPN(id, features, feature_tds, depth_idx,
                                     P6_in, P7_in, next_input, next_td):
    """Computes next input feature for upward propagation.

    # Arguments
        id :Int, the ID or index of the BiFPN block.
        features :List, the features returned from EfficientNet
                  backbone.
        feature_tds :Tensor, the feature resulting for upward
                     or downward propagation.
        depth_idx :Int, the depth of the feature of BiFPN layer.
        P6_in :Tensor, the output tensor from the P6 layer
               of EfficientNet.
        P7_in :Tensor, the output tensor from the P7 layer
               of EfficientNet.
        next_input :Tensor, the feature tensor from the relatively
                    top layer.
        next_td :Tensor, the feature tensor from the relatively
                    top layer as result of upward or downward
                    propagation.

    # Returns
        next_input :Tensor, the next input feature for upward
                    propagation.
        next_td :Tensor, the next input feature for upward
                    propagation generated from previous iteration of
                    upward propagation.
    """
    is_non_repeating_block = id == 0

    if is_non_repeating_block:
        next_input = {1: features[-1], 2: P6_in, 3: P7_in, 4: None}
        return next_input[depth_idx], feature_tds[2 - depth_idx]

    else:
        is_layer_not_P4 = depth_idx < len(features) - 2
        if is_layer_not_P4:
            next_input = features[2 + depth_idx]
            next_td = feature_tds[-3 - depth_idx]

        return next_input, next_td
