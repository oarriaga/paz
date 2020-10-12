import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input


class Loss():
    """Computes loss for Mask RCNN architecture

    # Arguments:
        RPN_output: [rpn_class_logits, rpn_bbox]
        target: [target_class_ids, target_boxes, target_masks]
        predictions: [class_logits, boxes, masks] from FPN head
        active_class_ids: List of class_ids available in the dataset
    """
    def __init__(self, config, RPN_output, target, predictions,
                 active_class_ids):
        self.config = config
        self.RPN_class_logits, self.RPN_boxes = RPN_output
        self.target = target
        self.predictions = predictions
        self.active_class_ids = active_class_ids
        self.rpn_match = Input(
                shape=[None, 1], name='input_rpn_match', dtype=tf.int32)
        self.input_RPN_box = Input(
            shape=[None, 4], name='input_rpn_bbox', dtype=tf.float32)

    def compute_loss(self):
        RPN_loss = [self.rpn_class_loss_graph(), self.rpn_bbox_loss_graph()]
        mrcnn_loss = [self.mrcnn_class_loss_graph(),
                      self.mrcnn_bbox_loss_graph(),
                      self.mrcnn_mask_loss_graph()]
        return RPN_loss + mrcnn_loss

    def batch_pack_graph(self, x, counts, num_rows):
        outputs = []
        for row in range(num_rows):
            outputs.append(x[row, :counts[row]])
        return tf.concat(outputs, axis=0)

    def smooth_L1_loss(self, y_true, y_pred):
        diff = K.abs(y_true - y_pred)
        less_than_one = K.cast(K.less(diff, 1.0), 'float32')
        loss = (less_than_one * 0.5 * diff**2) +\
            (1 - less_than_one) * (diff - 0.5)
        return loss

    def rpn_class_loss_graph(self):
        rpn_match = tf.squeeze(self.rpn_match, -1)
        anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
        indices = tf.where(K.not_equal(rpn_match, 0))
        rpn_class_logits = tf.gather_nd(self.RPN_class_logits, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                                 output=rpn_class_logits,
                                                 from_logits=True)
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        return loss

    def rpn_bbox_loss_graph(self):
        rpn_match = K.squeeze(self.rpn_match, -1)
        indices = tf.where(K.equal(rpn_match, 1))
        rpn_bbox = tf.gather_nd(self.RPN_boxes, indices)
        batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
        target_boxes = self.batch_pack_graph(self.input_RPN_box, batch_counts,
                                             self.config.IMAGES_PER_GPU)
        loss = self.smooth_L1_loss(target_boxes, rpn_bbox)
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        return loss

    def mrcnn_class_loss_graph(self):
        target_class_ids, _, _ = self.target
        predicted_class_logits, _, _ = self.predictions
        target_class_ids = tf.cast(target_class_ids, 'int64')
        pred_class_ids = tf.argmax(predicted_class_logits, axis=2)
        pred_active = tf.gather(self.active_class_ids[0], pred_class_ids)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_class_ids, logits=predicted_class_logits)
        loss = loss * pred_active
        loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
        return loss

    def mrcnn_bbox_loss_graph(self):
        target_class_ids, target_boxes, _ = self.target
        _, predicted_boxes, _ = self.predictions
        target_class_ids = K.reshape(target_class_ids, (-1,))
        target_boxes = K.reshape(target_boxes, (-1, 4))
        predicted_boxes = K.reshape(predicted_boxes,
                                    (-1, K.int_shape(predicted_boxes)[2], 4))
        positive_ROI_indices = tf.where(target_class_ids > 0)[:, 0]
        positive_ROI_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_ROI_indices), tf.int64)
        indices = tf.stack([positive_ROI_indices, positive_ROI_class_ids],
                           axis=1)
        target_boxes = tf.gather(target_boxes, positive_ROI_indices)
        predicted_boxes = tf.gather_nd(predicted_boxes, indices)
        loss = K.switch(tf.size(target_boxes) > 0,
                        self.smooth_L1_loss(target_boxes, predicted_boxes),
                        tf.constant(0.0))
        loss = K.mean(loss)
        return loss

    def mrcnn_mask_loss_graph(self):
        target_class_ids, _, target_masks = self.target
        _, _, predicted_masks = self.predictions
        target_class_ids = K.reshape(target_class_ids, (-1,))
        mask_shape = tf.shape(target_masks)
        target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
        pred_shape = tf.shape(predicted_masks)
        predicted_masks = K.reshape(predicted_masks,
                                    (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
        predicted_masks = tf.transpose(predicted_masks, [0, 3, 1, 2])
        positive_indices = tf.where(target_class_ids > 0)[:, 0]
        positive_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_indices), tf.int64)
        indices = tf.stack([positive_indices, positive_class_ids], axis=1)
        y_true = tf.gather(target_masks, positive_indices)
        y_pred = tf.gather_nd(predicted_masks, indices)
        loss = K.switch(tf.size(y_true) > 0,
                        K.binary_crossentropy(target=y_true, output=y_pred),
                        tf.constant(0.0))
        loss = K.mean(loss)
        return loss
