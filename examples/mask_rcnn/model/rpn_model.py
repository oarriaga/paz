import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Dense, Activation
from tensorflow.keras.layers import Lambda, Input, Concatenate

from tensorflow.keras.models import Model


def rpn_model(rpn_anchor_stride, rpn_anchor_ratios, fpn_size, rpn_feature_maps):
    """Build complete region specific network.

    # Arguments:
        fpn_size
        rpn_feature_masks
    # Returns:
        Model output.
    """
    rpn = build_rpn_model(rpn_anchor_stride, len(rpn_anchor_ratios), fpn_size)
    layer_outputs = [rpn([feature]) for feature in rpn_feature_maps]
    names = ['rpn_class_logits', 'rpn_class', 'rpn_bounding_box']
    outputs = list(zip(*layer_outputs))
    outputs = [Concatenate(axis=1, name=name)(list(output))
               for output, name in zip(outputs, names)]

    return outputs


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.

    # Arguments:
        anchors_per_location: number of anchors per pixel in feature map
        anchor_stride: Anchor for every pixel in feature map
                       Typically 1 or 2
        depth: Depth of the backbone feature map.

    # Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2]
                          Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2]
                   Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location,
                  (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
    """
    input_feature_map = Input(shape=[None, None, depth],
                              name='input_rpn_feature_map')
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return Model([input_feature_map], outputs, name='rpn_model')


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    # Arguments:
        feature_map: backbone features [batch, height, width, depth]
        anchors_per_location: number of anchors per pixel in feature map
        anchor_stride: Typically 1 (anchors for every pixel in feature map),
                       or 2 (every other pixel).

    # Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2]
                           Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2]
                    Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors, (dy, dx, log(dh), log(dw))]
                   Deltas to be applied to anchors.
    """
    shared = Conv2D(512, (3, 3), padding='same', activation='relu',
                    strides=anchor_stride,
                    name='rpn_conv_shared')(feature_map)
    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
               activation='linear', name='rpn_class_raw')(shared)
    rpn_class_logits = Lambda(lambda t: tf.reshape(t,
                              [tf.shape(t)[0], -1, 2]))(x)
    rpn_probs = Activation('softmax', name='rpn_class_xxx')(rpn_class_logits)
    x = Conv2D(anchors_per_location * 4, (1, 1), padding='valid',
               activation='linear', name='rpn_bbox_pred')(shared)
    rpn_bbox = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    return [rpn_class_logits, rpn_probs, rpn_bbox]
