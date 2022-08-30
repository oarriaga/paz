import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class ProposalClassLoss(tf.keras.layers.Layer):
    """Computes loss for Mask RCNN architecture

    """
    def __init__(self, config, name= 'rpn_class_loss'):
        self.config = config
        super(ProposalClassLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, -1)
        anchor_class = K.cast(K.equal(y_true, 1), tf.int32)
        indices = tf.where(K.not_equal(y_true, 0))
        y_pred = tf.gather_nd(y_pred, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                                 output=y_pred,
                                                 from_logits=True)
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        self.add_loss(loss)
        metric = (tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'rpn_class_logits', 1.))
        self.add_metric(metric, name= 'rpn_class_logits', aggregation='mean')
        return loss


class ProposalBBoxLoss(tf.keras.layers.Layer):
    """Computes loss for Mask RCNN architecture

    """
    def __init__(self, config, name= 'rpn_bbox_loss', rpn_match= None):
        self.config = config
        self.rpn_match = rpn_match
        super(ProposalBBoxLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        rpn_match = K.squeeze(self.rpn_match, -1)
        indices = tf.where(K.equal(rpn_match, 1))
        rpn_bbox = tf.gather_nd(y_pred, indices)
        batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
        target_boxes = batch_pack_graph(y_true, batch_counts,
                                             self.config.IMAGES_PER_GPU)
        loss = smooth_L1_loss(target_boxes, rpn_bbox)
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        self.add_loss(loss)
        metric = (tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'rpn_class_logits', 1.))
        self.add_metric(metric, name='rpn_bbox', aggregation='mean')
        return  loss


class ClassLoss(tf.keras.layers.Layer):
    """Computes loss for Mask RCNN architecture

    """
    def __init__(self, config, name='mrcnn_class_loss', active_class_ids =None):
        self.config = config
        self.active_class_ids = active_class_ids
        super(ClassLoss, self).__init__(name=name)


    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'int64')
        pred_class_ids = tf.argmax(y_pred, axis=2)
        pred_active = tf.gather(self.active_class_ids, pred_class_ids)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)
        loss = tf.reshape(loss,[-1]) * tf.cast(pred_active, 'float32')
        loss = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(pred_active,'float32'))
        self.add_loss(loss)
        metric = (tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'rpn_class_logits', 1.))
        self.add_metric(metric, name='mrcnn_class', aggregation='mean')
        return loss


class BBoxLoss(tf.keras.layers.Layer):
    """Computes loss for Mask RCNN architecture

    """
    def __init__(self, config, name='mrcnn_bbox_loss', target_class_ids=None):
        self.config = config
        self.target_class_ids = target_class_ids
        super(BBoxLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        target_boxes = y_true
        target_class_ids = K.reshape(self.target_class_ids, (-1,))
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
        self.add_loss(loss)
        metric = (tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'rpn_class_logits', 1.))
        self.add_metric(metric, name='mrcnn_bbox', aggregation='mean')
        return loss


class MaskLoss(tf.keras.layers.Layer):  #TODO: Rename MaskLoss. All class names as well
    """Computes loss for Mask RCNN architecture

    """
    def __init__(self, config, name='mrcnn_mask_loss', target_class_ids=None):
        self.config = config
        self.target_class_ids = target_class_ids
        super(MaskLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        target_ids, true_masks, pred_masks = reshape_data(self.target_class_ids,
                                                               y_true, y_pred)
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
        self.add_loss(loss)
        metric = (tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'rpn_class_logits', 1.))
        self.add_metric(metric, name='mrcnn_mask', aggregation='mean')
        return loss


def smooth_L1_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), 'float32')
    loss = (less_than_one * 0.5 * diff**2) +\
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