import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Layer


class ProposalClassLoss(tf.keras.losses.Loss):
    """Computes loss for Mask RCNN architecture, Region Proposal
    Network Class loss. RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """

    def __init__(self, loss_weight=1.0, name='rpn_class_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, -1)
        anchor_class = K.cast(K.equal(y_true, 1), tf.int32)
        indices = tf.compat.v1.where(K.not_equal(y_true, 0))
        y_pred = tf.gather_nd(y_pred, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                                 output=y_pred,
                                                 from_logits=True)
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))

        # self.add_loss(tf.math.reduce_mean(loss, keepdims=True) * self.loss_weight)
        # metric = (loss * self.loss_weight)
        # self.add_metric(metric, name='rpn_class_loss', aggregation='mean')
        return loss


class ProposalBBoxLoss(tf.keras.losses.Loss):
    """Computes loss for Mask RCNN architecture for Region Proposal
     Network Bounding box loss.
     Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    def __init__(self, anchors_per_image, image_per_gpu, loss_weight=1.0, name='rpn_bbox_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.anchors_per_image = anchors_per_image
        self.images_per_gpu = image_per_gpu
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        input_rpn_bbox = y_true[1]  # y_true[:, :self.anchors_per_image, :]
        rpn_match = y_true[0]  # y_true[:, self.anchors_per_image:, :1]

        rpn_match = K.squeeze(rpn_match, -1)
        indices = tf.where(K.equal(rpn_match, 1))

        rpn_bbox = tf.gather_nd(y_pred, indices)
        batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
        target_boxes = batch_pack_graph(input_rpn_bbox, batch_counts,
                                        self.images_per_gpu)
        loss = smooth_L1_loss(target_boxes, rpn_bbox)
        loss = K.switch(tf.size(input=loss) > 0, K.mean(loss), tf.constant(0.0))

        # self.add_loss(tf.math.reduce_mean(loss, keepdims=True) * self.loss_weight)
        # metric = (loss * self.loss_weight)
        # self.add_metric(metric, name='rpn_bbox_loss', aggregation='mean')
        return loss


class ClassLoss(Layer):
    """Computes loss for Mask RCNN architecture, for MRCNN class loss
    Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """

    def __init__(self, num_classes, loss_weight=1.0, name='mrcnn_class_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.active_class_ids = tf.ones([num_classes], dtype=tf.int32)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'int64')
        pred_class_ids = tf.argmax(input=y_pred, axis=2)
        pred_active = tf.gather(self.active_class_ids, pred_class_ids)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)
        loss = loss * tf.cast(pred_active, 'float32')

        loss = tf.math.reduce_sum(loss) / (tf.math.reduce_sum(
            input_tensor=tf.cast(pred_active, 'float32')))

        self.add_loss(loss * self.loss_weight)
        metric = (loss * self.loss_weight)
        self.add_metric(metric, name='mrcnn_class_loss', aggregation='mean')
        return loss


class BBoxLoss(Layer):
    """Computes loss for Mask RCNN architecture, MRCNN BBox loss
    Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    def __init__(self, loss_weight=1.0, name='mrcnn_bbox_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        target_boxes = y_true[0]
        target_class_ids = y_true[1]
        target_class_ids = K.reshape(target_class_ids, (-1,))
        target_boxes = K.reshape(target_boxes, (-1, 4))
        predicted_boxes = K.reshape(y_pred,
                                    (-1, K.int_shape(y_pred)[2], 4))
        positive_ROI_indices = tf.where(target_class_ids > 0)[:, 0]
        positive_ROI_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_ROI_indices), tf.int64)
        indices = tf.stack([positive_ROI_indices, positive_ROI_class_ids],
                           axis=1)
        target_boxes = tf.gather(target_boxes, positive_ROI_indices)
        predicted_boxes = tf.gather_nd(predicted_boxes, indices)
        loss = K.switch(tf.size(target_boxes) > 0,
                        smooth_L1_loss(target_boxes, predicted_boxes),
                        tf.constant(0.0))
        loss = K.mean(loss)

        self.add_loss(loss * self.loss_weight)
        metric = (loss * self.loss_weight)
        self.add_metric(metric, name='mrcnn_bbox_loss', aggregation='mean')
        return loss


class MaskLoss(Layer):
    """Computes loss for Mask RCNN architecture, for MRCNN mask loss.
    Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """

    def __init__(self, loss_weight=1.0, name='mrcnn_mask_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        true_masks = y_true[0]
        target_class_ids = y_true[1]
        target_ids, true_masks, pred_masks = reshape_data(target_class_ids,
                                                          true_masks, y_pred)
        positive_indices = tf.where(target_ids > 0)[:, 0]
        positive_class_ids = tf.cast(
            tf.gather(target_ids, positive_indices), tf.int64)
        indices = tf.stack([positive_indices, positive_class_ids], axis=1)

        y_true = tf.gather(true_masks, positive_indices)
        y_pred = tf.gather_nd(pred_masks, indices)
        loss = K.switch(tf.size(y_true) > 0,
                        K.binary_crossentropy(target=y_true, output=y_pred),
                        tf.constant(0.0))
        loss = K.mean(loss)

        self.add_loss(loss * self.loss_weight)
        metric = (loss * self.loss_weight)
        self.add_metric(metric, name='mrcnn_mask_loss', aggregation='mean')
        return loss


def smooth_L1_loss(y_true, y_pred):
    diff = tf.math.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), 'float32')
    loss = (less_than_one * 0.5 * diff ** 2) + \
           (1 - less_than_one) * (diff - 0.5)
    return loss


def reshape_data(target_ids, target_masks, y_pred):
    target_ids = K.reshape(target_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks,
                             (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(y_pred)
    y_pred = K.reshape(y_pred,
                       (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    y_pred = tf.transpose(y_pred, [0, 3, 1, 2])
    return target_ids, target_masks, y_pred


def batch_pack_graph(boxes, counts, num_rows):
    outputs = []
    for row in range(num_rows):
        outputs.append(boxes[row, :counts[row]])
    return tf.concat(outputs, axis=0)
