import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Layer


class BoundingBoxLoss(Layer):
    """Computes loss for Mask RCNN architecture, MRCNN bounding Box loss
    Loss for Mask R-CNN bounding box refinement.

    # Arguments:
        y_true = [target_bounding_box, target_class_ids]
        y_pred: [pred_bounding_box]

    # Returns:
        loss: bounding box loss value
    """

    def __init__(self, loss_weight=1.0, name='mrcnn_bounding_box_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        target_class_ids, target_boxes, predicted_boxes = reshape_values(y_true, y_pred)

        loss = smooth_L1_loss(target_boxes, predicted_boxes)
        loss = K.switch(tf.size(target_boxes) > 0, loss,
                        tf.constant(0.0))

        loss = K.mean(loss)
        self.add_loss(loss * self.loss_weight)
        metric = (loss * self.loss_weight)
        self.add_metric(metric, name=name, aggregation='mean')
        return loss


def reshape_values(y_true, y_pred):
    """Reshapes the y_true and y_pred values to compute MRCNN bounding Box loss
    by gathering positive target values of bounding boxes and class ids .

    # Arguments:
        y_true = [target_bounding_box, target_class_ids]
        y_pred: [pred_bounding_box]

    # Returns:
        target_bounding_box: [batch, num_rois, (dx, dy, log(dx), log(dh))]
        target_class_ids: [batch, num_rois]. Integer class IDs.
        pred_bounding_box: [batch, num_rois, num_classes, (dx, dy, log(dw), log(dh))]
    """
    target_class_ids = K.reshape(y_true[1], (-1,))
    target_boxes = K.reshape(y_true[0], (-1, 4))
    positive_target_indices = tf.where(y_true[1] > 0)[:, 0]

    target_boxes = tf.gather(target_boxes, positive_target_indices)
    positive_target_class_ids = tf.gather(target_class_ids, positive_target_indices)
    positive_target_class_ids = tf.cast(positive_target_class_ids, tf.int64)

    indices = tf.stack([positive_target_indices, positive_target_class_ids], axis=1)

    predicted_boxes = K.reshape(y_pred, (-1, K.int_shape(y_pred)[2], 4))
    predicted_boxes = tf.gather_nd(predicted_boxes, indices)

    return target_class_ids, target_boxes, predicted_boxes


def smooth_L1_loss(y_true, y_pred):
    """Returns the smooth L1 loss from the y_true and y_pred values

    # Arguments:
        y_true = [N]
        y_pred = [N]

    # Returns:
        loss: smooth L1 loss
    """
    diff = tf.math.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), 'float32')
    loss = (less_than_one * 0.5 * diff ** 2) + \
           (1 - less_than_one) * (diff - 0.5)
    return loss
