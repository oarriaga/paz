import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Layer,
                                     MaxPooling2D, SeparableConv2D,
                                     UpSampling2D)

from utils import GetDropConnect


def ClassNet(features, num_classes=90, num_anchors=9, num_filters=32,
             min_level=3, max_level=7, repeats=4, survival_rate=None,
             return_base=False, name='class_net/'):
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
        survival_rate: Float. If a value is set then drop connect will
            be used.
        return_base: Bool. Build the base feature network only.
            Excluding final class head.
        name: String indicating the name of this layer.

    # Returns
        box_outputs: List, BoxNet output offsets for each level.
    """
    class_outputs = build_predictionnet(
        repeats, num_filters, name, min_level, max_level, num_classes,
        num_anchors, features, survival_rate, return_base, build_classnet=True)
    return class_outputs


def BoxNet(features, num_anchors=9, num_filters=32, min_level=3,
           max_level=7, repeats=4, survival_rate=None,
           return_base=False, name='box_net/'):
    """Initialize the BoxNet.

    # Arguments
        features: List, feature to be processed by BoxNet head.
        num_anchors: Integer. Number of anchors.
        num_filters: Integer. Number of filters for intermediate
            layers.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        repeats: Integer. Number of intermediate layers.
        survival_rate: Float. If a value is set then drop connect
            will be used.
        return_base: Bool. Build the base feature network only.
            Excluding final class head.
        name: String indicating the name of this layer.

    # Returns
        box_outputs: List, BoxNet output offsets for each level.
    """
    box_outputs = build_predictionnet(
        repeats, num_filters, name, min_level, max_level, None, num_anchors,
        features, survival_rate, return_base, build_classnet=False)
    return box_outputs


def efficientnet_to_BiFPN(features, num_filters, fusion):
    """Propagates featutures from EfficientNet backbone to the
    first BiFPN block.

    # Arguments
    features: List, feature to be processed by BiFPN.
    num_filters: Integer. Number of filters for intermediate layers.
    fusion: String representing the feature fusion method
        in BiFPN.

    # Returns
    output_features: List, features after BiFPN for the class and box heads.
    """
    P6_in, P7_in = preprocess_features_BiFPN(
        0, features[-1], num_filters, features, 0, True)
    past_feature_map, now_feature_map = P7_in, P6_in
    feature_down = propagate_downwards_BiFPN_non_repeated(
        features, past_feature_map, now_feature_map, fusion, num_filters, 0)
    now_feature_map = feature_down[3]
    next_feature_map, next_td = features[-2], feature_down[2]
    output_features = [feature_down[3]]
    output_features = propagate_upwards_BiFPN_non_repeated(
        features, now_feature_map, next_feature_map, next_td,
        feature_down, fusion, num_filters, output_features, P6_in, P7_in)
    return output_features


def BiFPN_to_BiFPN(features, num_filters, fusion, id):
    """Propagates featutures from earlier BiFPN blocks to
    succeeding BiFPN block.

    # Arguments
    features: List, feature to be processed by BiFPN.
    num_filters: Integer. Number of filters for intermediate layers.
    fusion: String representing the feature fusion method
        in BiFPN.
    id: Integer. Represents the BiFPN repetition count.

    # Returns
    output_features: List, features after BiFPN for the class and box heads.
    """
    past_feature_map, now_feature_map = features[-1], features[-2]
    feature_down = propagate_downwards_BiFPN_repeated(
        features, past_feature_map, now_feature_map, fusion, num_filters, id)
    now_feature_map = feature_down[-1]
    next_feature_map, next_td = features[1], feature_down[-2]
    output_features = [feature_down[-1]]
    output_features = propagate_upwards_BiFPN_repeated(
        features, now_feature_map, next_feature_map, next_td,
        id, feature_down, fusion, num_filters, output_features)
    return output_features


class FuseFeature(Layer):
    def __init__(self, name, fusion, **kwargs):
        super().__init__(name=name, **kwargs)
        self.fusion = fusion
        if fusion == 'fast':
            self.fuse_method = self._fuse_fast
        elif fusion == 'sum':
            self.fuse_method = self._fuse_sum
        else:
            raise ValueError('FPN weight fusion is not defined')

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(
            self.name, (num_in,), tf.float32,
            tf.keras.initializers.constant(1 / num_in), trainable=True)

    def call(self, inputs, fusion):
        """
        # Arguments
        inputs: Tensor. Features to be fused.
        fusion: String representing the feature fusion
            method.

        # Returns
        x: feature after combining by the feature fusion method in
           BiFPN.
        """
        return self.fuse_method(inputs)

    def _fuse_fast(self, inputs):
        w = tf.keras.activations.relu(self.w)

        pre_activations = []
        for input_arg in range(len(inputs)):
            pre_activations.append(w[input_arg] * inputs[input_arg])
        x = tf.reduce_sum(pre_activations, 0)
        x = x / (tf.reduce_sum(w) + 0.0001)
        return x

    def _fuse_sum(self, inputs):
        x = inputs[0]
        for node in inputs[1:]:
            x = x + node
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({'fusion': self.fusion})
        return config


def conv2D_layer(num_filters, kernel_size, padding, activation,
                 name, bias_initializer):
    """Gets the conv2D layer in ClassNet class.
    # Arguments
        num_filters: Integer. Number of intermediate layers.
        kernel_size: Integer. Kernel size of conv layer.
        padding: String. Padding for conv layer.
        activation: String. Activation function.
        name: String. Name of conv layer.
        bias_initializer: String or TF Function. Bias
            initialization.

    # Returns
        conv2D_layer: TF conv layer.
    """
    conv2D_layer = SeparableConv2D(
        num_filters, kernel_size, (1, 1), padding, 'channels_last',
        (1, 1), 1, activation, True, tf.initializers.variance_scaling(),
        tf.initializers.variance_scaling(), bias_initializer, name=name)
    return conv2D_layer


def build_predictionnet(repeats, num_filters, name, min_level, max_level,
                        num_classes, num_anchors, features, survival_rate,
                        return_base, build_classnet):
    """Builds Prediction Net part of the Efficientdet

    # Arguments
        repeats: List, feature to be processed by PredictionNet head.
        num_filters: Integer. Number of filters for intermediate
                     layers.
        with_separable_conv: Bool.
        name: String indicating the name of this layer.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        num_classes: Integer. Number of classes.
        num_anchors: Integer. Number of anchors.
        features: Tuple. Input features for PredictionNet
        survival_rate: Float. If a value is set then drop connect
            will be used.
        return_base: Bool. Build the base feature network only.
            Excluding final class head.
        build_classnet: Bool.

    # Returns
        predictor_outputs: List. Output of PredictionNet block.
    """
    conv_blocks = build_predictionnet_conv_blocks(
        repeats, num_filters, name, build_classnet)

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
        num_filters, 3, 'same', None, layer_name,
        bias_initializer)

    predictor_outputs = []
    for level_id in range(num_levels):
        level_feature_map = propagate_forward_predictionnet(
            features, level_id, repeats, conv_blocks, batchnorms,
            survival_rate, return_base, classes)
        predictor_outputs.append(level_feature_map)
    return predictor_outputs


def build_predictionnet_conv_blocks(repeats, num_filters, name, is_classnet):
    """Builds convolutional blocks for PredictionNet

    # Arguments
        repeats: List, feature to be processed by PredictionNet head.
        num_filters: Integer. Number of filters for intermediate
            layers.
        name: String indicating the name of this layer.
        is_classnet: Bool.

    # Returns
        conv_blocks: List. Convolutional blocks for PredictionNet.
    """
    layer_name_prefix = name + 'class' if is_classnet else name + 'box'
    conv_blocks = []
    for repeat_args in range(repeats):
        conv_blocks.append(conv2D_layer(
            num_filters, 3, 'same', None,
            layer_name_prefix + '-%d' % repeat_args, tf.zeros_initializer()))
    return conv_blocks


def build_predictionnet_batchnorm_blocks(repeats, min_level, max_level,
                                         name, is_classnet):
    """Builds batch normalization blocks for PredictionNet

    # Arguments
        repeats: List, feature to be processed by PredictionNet head.
        min_level: Integer. Minimum level for features.
        max_level: Integer. Maximum level for features.
        name: String indicating the name of this layer.
        is_classnet: Bool.

    # Returns
        batchnorms: List. Batch normalization blocks for PredictionNet.
    """
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
                                    batchnorms, survival_rate,
                                    return_base, output_candidates):
    """Propagates features through PredictionNet block.

    # Arguments
        features: Tuple. Input features for PredictionNet
        level_id: Int. The index of feature level.
        repeats: List, feature to be processed by PredictionNet head.
        conv_blocks: List. Convolutional blocks for PredictionNet.
        batchnorms: List. Batch normalization blocks for PredictionNet.
        survival_rate: Float. If a value is set then drop connect will
            be used.
        return_base: Bool. Build the base feature network only.
            Excluding final class head.
        output_candidates: Tensor. PredictionNet output for each level.

    # Returns
        level_feature: List. Batch normalization blocks for
                       PredictionNet.
    """
    level_feature_map = features[level_id]
    for repeat_args in range(repeats):
        original_level_feature_map = level_feature_map
        level_feature_map = conv_blocks[repeat_args](level_feature_map)
        level_feature_map = batchnorms[repeat_args][level_id](
            level_feature_map)
        level_feature_map = tf.nn.swish(level_feature_map)
        if repeat_args > 0 and survival_rate:
            level_feature_map = GetDropConnect(
                survival_rate=survival_rate)(level_feature_map)
            level_feature_map = level_feature_map + original_level_feature_map

    if return_base:
        return level_feature_map
    else:
        return output_candidates(level_feature_map)


def propagate_downwards_BiFPN_non_repeated(features, past_feature_map,
                                           now_feature_map, fusion,
                                           num_filters, id):
    """Propagates features in downward direction through the first
    or non-repeated BiFPN block.

    # Arguments
        features: Tuple. Input features for PredictionNet
        past_feature_map: Tensor, feature input from previous
            level.
        now_feature_map: Tensor, feature input from current level.
        fusion: String representing the feature fusion
            method.
        num_filters: Integer. Number of filters for intermediate
            layers.
        id: Int. Represents the BiFPN repetition count.

    # Returns
        feature_down: List. Output features resulting from
                     down propagation from BiFPN block.
    """
    feature_down = []
    for depth_arg in range(len(features) - 1):
        past_feature_map = propagate_downwards_one_step_BiFPN(
            past_feature_map, now_feature_map, id,
            fusion, depth_arg, features, num_filters)
        now_feature_map_arg = refer_next_input_non_repeated(depth_arg)
        now_feature_map = features[now_feature_map_arg]
        feature_down.append(past_feature_map)
    return feature_down


def propagate_downwards_BiFPN_repeated(features, past_feature_map,
                                       now_feature_map, fusion,
                                       num_filters, id):
    """Propagates features in downward direction through the
    repeated BiFPN block.

    # Arguments
        features: Tuple. Input features for PredictionNet
        past_feature_map: Tensor, feature input from previous
            level.
        now_feature_map: Tensor, feature input from current level.
        fusion: String representing the feature fusion
            method.
        num_filters: Integer. Number of filters for intermediate
            layers.
        id: Int. Represents the BiFPN repetition count.

    # Returns
        feature_down: List. Output features resulting from
                     down propagation from BiFPN block.
    """
    feature_down = [past_feature_map]
    for depth_arg in range(len(features) - 1):
        past_feature_map = propagate_downwards_one_step_BiFPN(
            past_feature_map, now_feature_map, id,
            fusion, depth_arg, features, num_filters)
        now_feature_map_arg = refer_next_input_repeated(depth_arg, features)
        now_feature_map = features[now_feature_map_arg]
        feature_down.append(past_feature_map)
    return feature_down


def propagate_downwards_one_step_BiFPN(past_feature_map, now_feature_map,
                                       id, fusion, depth_arg, features,
                                       num_filters):
    """Propagates features in downward direction starting from the
    features of top most layer of EfficientNet backbone.

    # Arguments
        past_feature_map :Tensor, feature from the relatively
            top layer.
        now_feature_map :Tensor, feature from the current layer.
        id :Int, the ID or index of the BiFPN block.
        fusion :string, String representing the feature
            fusion method.
        depth_arg :Int, the depth of the feature of BiFPN layer.
        features :List, the features returned from EfficientNet
            backbone.
        num_filters :Int, Number of filters for intermediate layers.

    # Returns
        now_feature_map_td: Tensor, tensor resulting from
            down propagation in BiFPN layer.
    """
    is_non_repeated_block = id == 0
    layer_not_P7 = depth_arg > 0
    if is_non_repeated_block and layer_not_P7:
        now_feature_map = preprocess_features_BiFPN(
            depth_arg, now_feature_map, num_filters, features, id, True)

    past_feature_map_U = UpSampling2D()(past_feature_map)
    now_feature_map_td = FuseFeature(
        name=(f'FPN_cells/cell_{id}/fnode{depth_arg}/add'),
        fusion=fusion)(
        [now_feature_map, past_feature_map_U], fusion)

    now_feature_map_td = tf.nn.swish(now_feature_map_td)
    now_feature_map_td = SeparableConv2D(
        num_filters, 3, 1, 'same', use_bias=True,
        name=(f'FPN_cells/cell_{id}/fnode{depth_arg}/'
              f'op_after_combine{len(features) + depth_arg}/conv'))(
            now_feature_map_td)

    now_feature_map_td = BatchNormalization(
        name=(f'FPN_cells/cell_{id}/fnode{depth_arg}/'
              f'op_after_combine{len(features) + depth_arg}/bn'))(
            now_feature_map_td)
    return now_feature_map_td


def refer_next_input_non_repeated(depth_arg):
    """Computes and returns the index of the next input feature to be
    fed into the BiFPN block.

    # Arguments
        depth_arg :Int, depth of the BiFPN block.

    # Returns
        next_feature_map_arg :Int. indicating the index of the output
        feature.
    """
    next_feature_map_arg = -1 - depth_arg
    return next_feature_map_arg


def refer_next_input_repeated(depth_arg, features):
    """Computes and returns the index of the next input feature to be
    fed into the BiFPN block.

    # Arguments
        depth_arg :Int, depth of the BiFPN block.
        features :Int, the ID or index of the BiFPN block.

    # Returns
        next_feature_map_arg :Int. indicating the index of the output
        feature.
    """
    num_BiFPN_upsamplers = 3
    next_feature_map_arg = len(features) - num_BiFPN_upsamplers - depth_arg
    return next_feature_map_arg


def propagate_upwards_BiFPN_non_repeated(features, now_feature_map,
                                         next_feature_map, next_td,
                                         feature_down, fusion, num_filters,
                                         output_features, P6_in, P7_in):
    """Propagates features in upward direction through the non-repeated
    BiFPN block.

    # Arguments
        features: Tuple. Input features for PredictionNet
        now_feature_map: Tensor, feature input from current level.
        next_feature_map: Tensor, feature input from next level.
        next_td: Tensor, feature input from next level.
        id: Int. Represents the BiFPN repetition count.
        feature_down: List. Output features resulting from
            down propagation from BiFPN block.
        fusion: String representing the feature fusion
            method.
        num_filters: Integer. Number of filters for intermediate
            layers.
        output_features: List. Output features resulting from
            down propagation from BiFPN block.
        P6_in: Tensor, feature input from 6th level.
        P7_in: Tensor, feature input from 7th level.

    # Returns
        output_features: List. Output features resulting from
                         upward propagation from BiFPN block.
    """
    for depth_arg in range(len(features) - 1):
        now_feature_map = propagate_upwards_one_step_BiFPN_non_repeated(
            now_feature_map, next_feature_map, next_td, feature_down,
            depth_arg, fusion, num_filters, features)
        output_features.append(now_feature_map)
        depth_arg_next, P6_in_arg, P7_in_arg = depth_arg + 1, P6_in, P7_in
        next_feature_map_arg = None
        next_feature_map, next_td = compute_next_input_BiFPN_non_repeated(
            features, feature_down, depth_arg_next,
            P6_in_arg, P7_in_arg, next_feature_map_arg)
    return output_features


def propagate_upwards_BiFPN_repeated(features, now_feature_map,
                                     next_feature_map, next_td, id,
                                     feature_down, fusion, num_filters,
                                     output_features):
    """Propagates features in upward direction through the repeated
    BiFPN block.

    # Arguments
        features: Tuple. Input features for PredictionNet
        now_feature_map: Tensor, feature input from current level.
        next_feature_map: Tensor, feature input from next level.
        next_td: Tensor, feature input from next level.
        id: Int. Represents the BiFPN repetition count.
        feature_down: List. Output features resulting from
            down propagation from BiFPN block.
        fusion: String representing the feature fusion
            method.
        num_filters: Integer. Number of filters for intermediate
            layers.
        output_features: List. Output features resulting from
            down propagation from BiFPN block.

    # Returns
        output_features: List. Output features resulting from
            upward propagation from BiFPN block.
    """
    for depth_arg in range(len(features) - 1):
        now_feature_map = propagate_upwards_one_step_BiFPN_repeated(
            now_feature_map, next_feature_map, next_td, id, feature_down,
            depth_arg, fusion, num_filters, features)
        output_features.append(now_feature_map)
        depth_arg_next = depth_arg
        next_feature_map_arg, next_td_arg = next_feature_map, next_td
        next_feature_map, next_td = compute_next_input_feature_BiFPN_repeated(
            features, feature_down, depth_arg_next,
            next_feature_map_arg, next_td_arg)
    return output_features


def propagate_upwards_one_step_BiFPN_non_repeated(now_feature_map,
                                                  next_feature_map,
                                                  next_td, feature_down,
                                                  depth_arg, fusion,
                                                  num_filters, features):
    """Propagates features in upward direction in the BiFPN non
    repeated block starting from the features of bottom most layer
    of EfficientNet backbone.

    # Arguments
        now_feature_map :Tensor, Tensor, feature from the
            current layer.
        next_feature_map :Tensor, Tensor, feature from the relatively
            top layer.
        next_td : Tensor, The feature tensor from the relatively
            top layer as result of upward or downward propagation.
        id :Int, the ID or index of the BiFPN block.
        feature_down: List, the list of features as a result of
            upward or downward propagation.
        depth_arg :Int, the depth of the feature of BiFPN layer.
        fusion :string, String representing the feature
            fusion method.
        num_filters :Int, Number of filters for intermediate layers.
        features :List, the features returned from EfficientNet
            backbone.

    # Returns
        now_feature_td :Tensor, Tensor, tensor resulting from
            upward propagation in BiFPN layer.
    """
    id = 0
    now_feature_map_D = MaxPooling2D(3, 2, 'same')(now_feature_map)

    is_layer_P6_or_P7 = depth_arg < 2
    is_layer_P4 = depth_arg == 3

    if is_layer_P6_or_P7:
        next_feature_map = preprocess_features_BiFPN(
            depth_arg, next_feature_map, num_filters, features, id, False)

    layer_names = [(f'FPN_cells/cell_{id}/fnode'
                    f'{len(features) - 2 + depth_arg + 1}'
                    f'/add'),
                   (f'FPN_cells/cell_{id}/fnode'
                    f'{len(feature_down) + depth_arg}'
                    f'/op_after_combine{9 + depth_arg}'
                    f'/conv'),
                   (f'FPN_cells/cell_{id}/fnode'
                    f'{len(feature_down) + depth_arg}/'
                    f'op_after_combine{9 + depth_arg}'
                    f'/bn')]

    if is_layer_P4:
        to_fuse = [next_feature_map, now_feature_map_D]
    else:
        to_fuse = [next_feature_map, next_td, now_feature_map_D]

    next_out = FuseFeature(
        name=layer_names[0], fusion=fusion)(
        to_fuse, fusion)
    next_out = tf.nn.swish(next_out)
    next_out = SeparableConv2D(num_filters, 3, 1, 'same', use_bias=True,
                               name=layer_names[1])(next_out)
    next_out = BatchNormalization(name=layer_names[2])(next_out)
    return next_out


def propagate_upwards_one_step_BiFPN_repeated(now_feature_map,
                                              next_feature_map,
                                              next_td, id, feature_down,
                                              depth_arg, fusion,
                                              num_filters, features):
    """Propagates features in upward direction in the BiFPN repeated
    blockstarting from the features of bottom most layer of
    EfficientNet backbone.

    # Arguments
        now_feature_map :Tensor, Tensor, feature from the
            current layer.
        next_feature_map :Tensor, Tensor, feature from the relatively
            top layer.
        next_td : Tensor, The feature tensor from the relatively
            top layer as result of upward or downward propagation.
        id :Int, the ID or index of the BiFPN block.
        feature_down: List, the list of features as a result of
            upward or downward propagation.
        depth_arg :Int, the depth of the feature of BiFPN layer.
        fusion :string, String representing the feature
            fusion method.
        num_filters :Int, Number of filters for intermediate layers.
        features :List, the features returned from EfficientNet
            backbone.

    # Returns
        now_feature_td :Tensor, Tensor, tensor resulting from
            upward propagation in BiFPN layer.
    """
    now_feature_map_D = MaxPooling2D(3, 2, 'same')(now_feature_map)

    is_layer_P4 = depth_arg == 3

    layer_names = [(f'FPN_cells/cell_{id}/'
                    f'fnode{len(feature_down) - 1 + depth_arg}'
                    f'/add'),
                   (f'FPN_cells/cell_{id}/fnode'
                    f'{len(feature_down) - 1 + depth_arg}'
                    f'/op_after_combine'
                    f'%d' % (len(feature_down) + depth_arg +
                             len(features) - 2 + 1) + '/conv'),
                   (f'FPN_cells/cell_{id}/fnode'
                    f'{len(feature_down) - 1 + depth_arg}/'
                    f'op_after_combine'
                    f'%d' % (len(feature_down) + depth_arg +
                             len(features) - 2 + 1) + '/bn')]

    if is_layer_P4:
        to_fuse = [next_feature_map, now_feature_map_D]
    else:
        to_fuse = [next_feature_map, next_td, now_feature_map_D]

    next_out = FuseFeature(
        name=layer_names[0], fusion=fusion)(
        to_fuse, fusion)
    next_out = tf.nn.swish(next_out)
    next_out = SeparableConv2D(num_filters, 3, 1, 'same', use_bias=True,
                               name=layer_names[1])(next_out)
    next_out = BatchNormalization(name=layer_names[2])(next_out)
    return next_out


def preprocess_features_BiFPN(depth_arg, input_feature, num_filters,
                              features, id, is_propagate_downwards):
    """Perform pre-processing on features before applying
    downward propagation or upward propagation.

    # Arguments
        depth_arg :Int, the depth of the feature of BiFPN layer.
        input_feature :Tensor, feature from the current layer.
        num_filters :Int, Number of filters for intermediate layers.
        features :List, the features returned from EfficientNet
            backbone.
        id :Int, the ID or index of the BiFPN block.
        is_propagate_downwards :Bool, Boolean flag indicating if
            propagation is in upward or downward direction.

    # Returns
        preprocessed_feature: Tensor, the preprocessed feature.
    """
    is_layer_P7 = depth_arg == 0

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
            layer_names = [(f'FPN_cells/cell_{id}/fnode{depth_arg}/resample_0_'
                            f'{3 - depth_arg}_{len(features) + depth_arg}'
                            f'/conv2d'),
                           (f'FPN_cells/cell_{id}/fnode{depth_arg}/resample_0_'
                            f'{3 - depth_arg}_{len(features) + depth_arg}/bn')]
            preprocessed_feature = preprocess_features_partly_BiFPN(
                input_feature, num_filters, layer_names)
            return preprocessed_feature
    else:
        layer_names = [(f'FPN_cells/cell_{id}/fnode' +
                        '%d' % (len(features) - 1 + depth_arg) +
                        f'/resample_0_{1 + depth_arg}_{9 + depth_arg}/conv2d'),
                       (f'FPN_cells/cell_{id}'
                        f'/fnode{len(features) - 1 + depth_arg}'
                        f'/resample_0_{1 + depth_arg}_{9 + depth_arg}/bn')]
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


def compute_next_input_BiFPN_non_repeated(features, feature_down, depth_arg,
                                          P6_in, P7_in, next_feature_map):
    """Computes next input feature for upward propagation.

    # Arguments
        features :List, the features returned from EfficientNet
            backbone.
        feature_down :Tensor, the feature resulting for upward
            or downward propagation.
        depth_arg :Int, the depth of the feature of BiFPN layer.
        P6_in :Tensor, the output tensor from the P6 layer
            of EfficientNet.
        P7_in :Tensor, the output tensor from the P7 layer
            of EfficientNet.
        next_feature_map :Tensor, the feature tensor from the relatively
            top layer.

    # Returns
        next_input :Tensor, the next input feature for upward
            propagation.
        next_td :Tensor, the next input feature for upward
            propagation generated from previous iteration of
            upward propagation.
    """
    feature_offset_arg = 2
    next_feature_map = {1: features[-1], 2: P6_in, 3: P7_in, 4: None}
    return next_feature_map[depth_arg], feature_down[
        feature_offset_arg - depth_arg]


def compute_next_input_feature_BiFPN_repeated(features, feature_down,
                                              depth_arg, next_feature_map,
                                              next_td):
    """Computes next input feature for upward propagation.

    # Arguments
        features :List, the features returned from EfficientNet
            backbone.
        feature_down :Tensor, the feature resulting for upward
            or downward propagation.
        depth_arg :Int, the depth of the feature of BiFPN layer.
        next_feature_map :Tensor, the feature tensor from the relatively
            top layer.
        next_td :Tensor, the feature tensor from the relatively
            top layer as result of upward or downward propagation.

    # Returns
        next_input :Tensor, the next input feature for upward
            propagation.
        next_td :Tensor, the next input feature for upward
            propagation generated from previous iteration of
            upward propagation.
    """
    num_BiFPN_upsamplers = 3
    feature_offset_arg = 2
    is_layer_not_P4 = depth_arg < len(features) - feature_offset_arg
    if is_layer_not_P4:
        next_feature_map = features[feature_offset_arg + depth_arg]
        next_td = feature_down[
            -num_BiFPN_upsamplers - depth_arg]
    return next_feature_map, next_td
