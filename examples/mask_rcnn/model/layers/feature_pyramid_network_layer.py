import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Dense, Activation
from tensorflow.keras.layers import TimeDistributed, Lambda, Reshape
from tensorflow.keras.layers import Conv2DTranspose

from mask_rcnn.model.layers.pyramid_ROI_align import PyramidROIAlign
from tensorflow.keras.layers import BatchNormalization as BatchNorm


def fpn_classifier_graph(rois, feature_maps, num_classes, image_shape, train_bn=False,
                         fc_layers_size=1024, pool_size=7):
    """Builds the computation graph of the feature pyramid network classifier
       and regressor heads.

    # Arguments:
        rois: [batch, num_rois, (y_min, x_min, y_max, x_max)]
              Proposal boxes in normalized coordinates.
        feature_maps: List of feature maps from different pyramid layers,
                      [P2, P3, P4, P5].
        image_meta: [batch, (meta data)] Image details
        pool_size: The width of the square feature map generated from ROI Pool.
        num_classes: number of classes
        train_bn: Boolean. Train or freeze Batch Norm layers
        fc_layers_size: Size of the 2 FC layers

    # Returns:
        logits: classifier logits (before softmax)
                [batch, num_rois, NUM_CLASSES]
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bounding_box_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                     Deltas to apply to proposal boxes
    """
    x = PyramidROIAlign([pool_size, pool_size], name='roi_align_classifier')(
        [rois, (image_shape[0], image_shape[1])] + feature_maps)

    conv_2d_layer = Conv2D(fc_layers_size, (pool_size, pool_size),   # Rename fc_layers_size to fc_dim, fc check
                           padding='valid')                          # Fit everything in a single line

    x = TimeDistributed(conv_2d_layer, name='mrcnn_class_conv1')(x)
    x = TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(
        x, training=train_bn)
    x = Activation('relu')(x)       # Check where it is applied, axis
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)),
                        name='mrcnn_class_conv2')(x)
    x = TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(
        x, training=train_bn)
    x = Activation('relu')(x)
    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                    name='pool_squeeze')(x)
    # Classifier head
    mrcnn_class_logits = TimeDistributed(Dense(num_classes),
                                         name='mrcnn_class_logits')(shared)
    mrcnn_probs = TimeDistributed(Activation('softmax'),
                                  name='mrcnn_class')(mrcnn_class_logits)
    # Bounding box head
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'),
                        name='mrcnn_bbox_fc')(shared)
    s = K.int_shape(x)        # s rename, box_shape

    if s[1] is None:
        mrcnn_bbox = Reshape((-1, num_classes, 4), name='mrcnn_bbox')(x)
    else:
        mrcnn_bbox = Reshape((s[1], num_classes, 4), name='mrcnn_bbox')(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, num_classes, image_shape, train_bn=False, pool_size=14):
    """Builds computation graph of the mask head of Feature Pyramid Network.

    # Arguments:
        rois: [batch, num_rois, (y_min, x_min, y_max, x_max)]
              Proposal boxes in normalized coordinates.
        feature_maps: List of feature maps from different pyramid layers,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details
        pool_size: The width of the square feature map generated from ROI Pool.
        num_classes: number of classes
        train_bn: Boolean. Train or freeze Batch Norm layers

    # Returns:
        Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    x = PyramidROIAlign([pool_size, pool_size], name='roi_align_mask')(
        [rois, (image_shape[0], image_shape[1], image_shape[2])] + feature_maps)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                        name='mrcnn_mask_conv1')(x)
    x = TimeDistributed(BatchNorm(),
                        name='mrcnn_mask_bn1')(x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                        name='mrcnn_mask_conv2')(x)
    x = TimeDistributed(BatchNorm(),
                        name='mrcnn_mask_bn2')(x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                        name='mrcnn_mask_conv3')(x)
    x = TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(
        x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                        name='mrcnn_mask_conv4')(x)
    x = TimeDistributed(BatchNorm(),
                        name='mrcnn_mask_bn4')(x, training=train_bn)

    x = Activation('relu')(x)
    x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2,
                        activation='relu'), name='mrcnn_mask_deconv')(x)
    x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1,
                        activation='sigmoid'), name='mrcnn_mask')(x)
    return x
