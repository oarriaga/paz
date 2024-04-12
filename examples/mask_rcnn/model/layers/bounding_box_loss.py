import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class BoundingBoxLoss(Layer):
    """Computes loss for Mask RCNN architecture, MRCNN bounding Box loss
    Loss for Mask R-CNN bounding box refinement.

    # Arguments:
        y_true = [target_bounding_box, target_class_ids]
        y_pred: [pred_bounding_box]

    # Returns:
        loss: bounding box loss value
    """

    def __init__(self, loss_weight=1.0, name='mrcnn_bounding_box_loss',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        target_class_ids, positive_target_boxes = resize_targets(y_true)

        (target_class_ids, positive_target_boxes,
         predicted_positive_indices) = gather_all_positive_targets(
            target_class_ids, positive_target_boxes)

        predicted_boxes = gather_all_positive_predictions(
            predicted_positive_indices, y_pred)

        loss = smooth_L1_loss(positive_target_boxes, predicted_boxes)
        loss = K.switch(tf.size(positive_target_boxes) > 0, loss,
                        tf.constant(0.0))

        loss = K.mean(loss)
        self.add_loss(loss * self.loss_weight)
        metric = (loss * self.loss_weight)
        self.add_metric(metric, name='mrcnn_bounding_box_loss',
                        aggregation='mean')
        return loss


def resize_targets(y_true):
    """Reshapes the y_true values to compute M-RCNN bounding Box loss.

        # Arguments:
            y_true = [target_bounding_box, target_class_ids]

        # Returns:
            target_bounding_box: [batch, num_rois, (dx, dy, log(dx), log(dh))]
            target_class_ids: [batch, num_rois]. Integer class IDs.
        """
    target_class_ids = K.reshape(y_true[1], (-1,))
    positive_target_boxes = K.reshape(y_true[0], (-1, 4))
    return target_class_ids, positive_target_boxes


def gather_all_positive_targets(target_class_ids, positive_target_boxes):
    """Gathers positive target values of bounding boxes and class ids .

    # Arguments:
        target_bounding_box: [batch, num_rois, (dx, dy, log(dx), log(dh))]
        target_class_ids: [batch, num_rois]. Integer class IDs.

    # Returns:
        target_bounding_box: [batch, posiive_rois, (dx, dy, log(dx), log(dh))]
        target_class_ids: [batch, positive_rois]. Integer class IDs.
    """
    positive_target_indices = tf.where(target_class_ids > 0)[:, 0]
    positive_target_boxes = tf.gather(positive_target_boxes,
                                      positive_target_indices)
    positive_target_class_ids = tf.gather(target_class_ids,
                                          positive_target_indices)
    positive_target_class_ids = tf.cast(positive_target_class_ids, tf.int64)

    predicted_positive_indices = tf.stack([positive_target_indices,
                                           positive_target_class_ids], axis=1)
    return target_class_ids, positive_target_boxes, predicted_positive_indices


def gather_all_positive_predictions(predicted_positive_indices, y_pred):
    """Gathers all positive predictions from all predicted values.

        # Arguments:
            predicted_positive_indices: indices of positive predicted values
            y_pred: [pred_bounding_box]

        # Returns:
            predicted_boxes: [batch, num_rois, num_classes,
                               (dx, dy, log(dw), log(dh))]
        """
    predicted_boxes = K.reshape(y_pred, (-1, K.int_shape(y_pred)[2], 4))
    predicted_boxes = tf.gather_nd(predicted_boxes, predicted_positive_indices)
    return predicted_boxes


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
    loss = (less_than_one * 0.5 * diff ** 2)
    loss = loss + (1 - less_than_one) * (diff - 0.5)
    return loss
