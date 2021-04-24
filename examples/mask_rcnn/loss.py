import tensorflow as tf
import tensorflow.keras.backend as K


class Loss():
    """Computes loss for Mask RCNN architecture

    # Arguments:
        RPN_output: [rpn_class_logits, rpn_bbox]
        target: [target_class_ids, target_boxes, target_masks]
        predictions: [class_logits, boxes, masks] from FPN head
        active_class_ids: List of class_ids available in the dataset
    """
    def __init__(self, config):
        self.config = config

    def batch_pack_graph(self, boxes, counts, num_rows):
        outputs = []
        for row in range(num_rows):
            outputs.append(boxes[row, :counts[row]])
        return tf.concat(outputs, axis=0)

    def smooth_L1_loss(self, y_true, y_pred):
        diff = K.abs(y_true - y_pred)
        less_than_one = K.cast(K.less(diff, 1.0), 'float32')
        loss = (less_than_one * 0.5 * diff**2) +\
            (1 - less_than_one) * (diff - 0.5)
        return loss

    def rpn_class_loss_graph(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, -1)
        anchor_class = K.cast(K.equal(y_true, 1), tf.int32)
        indices = tf.where(K.not_equal(y_true, 0))
        y_pred = tf.gather_nd(y_pred, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                                 output=y_pred,
                                                 from_logits=True)
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        return loss

    def rpn_class_metrics(self, y_true, y_pred):
        loss = self.rpn_class_loss_graph(y_true, y_pred)
        metric = tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'rpn_class_logits', 1.)
        return metric

    def rpn_bbox_loss_graph(self, rpn_match, y_true, y_pred):
        rpn_match = K.squeeze(rpn_match, -1)
        indices = tf.where(K.equal(rpn_match, 1))
        rpn_bbox = tf.gather_nd(y_pred, indices)
        batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
        target_boxes = self.batch_pack_graph(y_true, batch_counts,
                                             self.config.IMAGES_PER_GPU)
        loss = self.smooth_L1_loss(target_boxes, rpn_bbox)
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        return loss

    def rpn_bbox_metrics(self, y_true, y_pred):
        loss = self.rpn_bbox_loss_graph(y_true, y_pred)
        metric = tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'rpn_bbox', 1.)
        return metric

    def mrcnn_loss_graph(self, y_true, y_pred):
        class_loss = self.mrcnn_class_loss_graph(y_true, y_pred)
        bbox_loss = self.mrcnn_bbox_loss_graph(y_true, y_pred)
        mask_loss = self.mrcnn_mask_loss_graph(y_true, y_pred)
        return class_loss + bbox_loss + mask_loss

    def mrcnn_metrics(self, y_true, y_pred):
        loss = self.mrcnn_loss_graph(y_true, y_pred)
        metric = tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'mrcnn', 1.)
        return metric

    def mrcnn_class_loss_graph(self, active_ids, y_true, y_pred):
        y_true = tf.cast(y_true, 'int64')
        pred_class_ids = tf.argmax(y_pred, axis=2)
        pred_active = tf.gather(active_ids, pred_class_ids)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)
        loss = loss * pred_active
        loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
        return loss

    def mrcnn_class_metrics(self, y_true, y_pred):
        loss = self.mrcnn_class_loss_graph(y_true, y_pred)
        metric = tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'mrcnn_class', 1.)
        return metric

    def mrcnn_bbox_loss_graph(self, y_true, y_pred):
        target_class_ids, target_boxes, _ = y_true
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
                        self.smooth_L1_loss(target_boxes, predicted_boxes),
                        tf.constant(0.0))
        loss = K.mean(loss)
        return loss

    def mrcnn_bbox_metrics(self, y_true, y_pred):
        loss = self.mrcnn_bbox_loss_graph(y_true, y_pred)
        metric = tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'mrcnn_bbox', 1.)
        return metric

    def mrcnn_mask_loss_graph(self, y_true, y_pred):
        target_ids, true_masks, pred_masks = self.reshape_data(y_true, y_pred)
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
        return loss

    def reshape_data(self, y_true, y_pred):
        target_ids, _, target_masks = y_true
        target_ids = K.reshape(target_ids, (-1,))
        mask_shape = tf.shape(target_masks)
        target_masks = K.reshape(target_masks,
                                 (-1, mask_shape[2], mask_shape[3]))
        pred_shape = tf.shape(y_pred)
        y_pred = K.reshape(y_pred,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
        y_pred = tf.transpose(y_pred, [0, 3, 1, 2])
        return target_ids, target_masks, y_pred

    def mrcnn_mask_metrics(self, y_true, y_pred):
        loss = self.mrcnn_mask_loss_graph(y_true, y_pred)
        metric = tf.reduce_mean(loss) * self.config.LOSS_WEIGHTS.get(
            'mrcnn_mask', 1.)
        return metric
