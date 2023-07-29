import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Dense, Activation
from tensorflow.keras.layers import Lambda, Input, Concatenate

from tensorflow.keras.models import Model


def RPN_model(RPN_anchor_stride, RPN_anchor_ratios, FPN_size,
              RPN_feature_maps):
    """Build complete region specific network.

    # Arguments:
        RPN_anchor_stride: Anchor stride. If 1 then anchors are created for
                           each cell in the backbone feature map. If 2, then
                           anchors are created for every other cell, and so on.
        RPN_anchor_ratios: Ratios of anchors at each cell (width/height). A
                           value of 1 represents a square anchor, and 0.5 is
                           a wide anchor
        FPN_size: Size of the fully-connected layers in the classification
                  graph.
        RPN_feature_maps: feature map of region proposal network.
    # Returns:
        Model output.
    """
    RPN = build_RPN(RPN_anchor_stride, len(RPN_anchor_ratios), FPN_size)
    layer_outputs = [RPN([feature]) for feature in RPN_feature_maps]
    names = ['rpn_class_logits', 'rpn_class', 'rpn_bounding_box']
    outputs = list(zip(*layer_outputs))
    outputs = [Concatenate(axis=1, name=name)(list(output))
               for output, name in zip(outputs, names)]
    return outputs


def build_RPN(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.

    # Arguments:
        anchors_per_location: number of anchors per pixel in feature map
        anchor_stride: Anchor for every pixel in feature map
                       Typically 1 or 2
        depth: Depth of the backbone feature map.

    # Returns:
        RPN_class_logits: [batch, H * W * anchors_per_location, 2]
                          Anchor classifier logits (before softmax)
        RPN_probs: [batch, H * W * anchors_per_location, 2]
                   Anchor classifier probabilities.
        RPN_bbox: [batch, H * W * anchors_per_location,
                  (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
    """
    feature_map = Input(shape=[None, None, depth],
                        name='input_rpn_feature_map')

    shared = Conv2D(512, (3, 3), padding='same', activation='relu',
                    strides=anchor_stride, name='rpn_conv_shared')(feature_map)

    RPN_class_logits = build_head(shared, 2, anchors_per_location, (1, 1),
                                  name='rpn_class_raw')
    RPN_probs = Activation('softmax', name='RPN_class_xxx')(RPN_class_logits)
    RPN_bbox = build_head(shared, 4, anchors_per_location, (1, 1),
                          name='rpn_bbox_pred')
    return Model([feature_map], [RPN_class_logits, RPN_probs, RPN_bbox],
                 name='rpn_model')


def build_head(x, num_dim, anchors_per_location, shape, name=''):
    """Builds the computation head for outputs of Region Proposal Network.

    # Arguments:
        x: backbone features.
        num_dim: number of anchors per pixel in feature map.
        anchor_per location: number of anchors per pixel in feature map.
        shape: size of the feature maps.

    # Returns:
        output: computational head
    """
    kwargs = {'padding': 'valid', 'activation': 'linear', 'name': name}
    x = Conv2D(num_dim * anchors_per_location, shape, **kwargs)(x)
    output = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, num_dim]))(x)
    return output
